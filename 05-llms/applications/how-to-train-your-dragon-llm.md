# How LLMs Actually Work

---

## The Decoder-Only Transformer

**The problem:** a model that can generate fluent text must produce one token at a time, with each token conditioned on all preceding tokens. Encoder-decoder architectures (BERT-style encoders, seq2seq models) require a separate encoding step and a different attention pattern for generation — overhead that is unnecessary when generation is the only task. Full bidirectional attention cannot be used for generation because it would allow each position to attend to future positions that have not yet been generated.

**The core insight:** a single stack of transformer blocks with causal masking — each position attends only to itself and positions to its left — unifies training and generation in one architecture. During training, all positions compute their predictions in parallel (no sequential dependency). During inference, generation proceeds left to right, one token per step.

**The mechanics:**

```
Input tokens → Token Embedding + Positional Encoding
                        │
               ┌────────▼─────────┐
               │  Decoder Block   │ × L
               │  ┌────────────┐  │
               │  │ LayerNorm  │  │
               │  │ Causal MHA │  │
               │  │ Residual   │  │
               │  └────────────┘  │
               │  ┌────────────┐  │
               │  │ LayerNorm  │  │
               │  │    FFN     │  │
               │  │ Residual   │  │
               │  └────────────┘  │
               └──────────────────┘
                        │
                   LayerNorm
                        │
              LM Head (Linear → Softmax)
                        │
                Token probabilities
```

Pre-norm (LayerNorm before the sublayer, not after) is standard in modern LLMs — it stabilizes training and allows larger learning rates than post-norm.

**What breaks:** causal masking makes encoder-style bidirectional representation impossible. For classification or retrieval tasks where the full context is available upfront, encoder models (BERT) have an information advantage. Decoder-only models compensate by using the full context in their key-value computations, but the causal mask structurally prevents left positions from attending to right positions.

---

## Tokenization (BPE)

**The problem:** a neural network cannot process raw text characters — or if it does, English has ~200 characters but vocabulary coverage requires thousands of distinct symbols, and Chinese requires tens of thousands. A character-level vocabulary is too small to represent meaningful units efficiently. A word-level vocabulary has hundreds of thousands of entries, cannot handle rare words, and fails entirely on subword morphology (run, running, runner all share a stem the model should recognize).

**The core insight:** the optimal granularity is between characters and words — subword units that appear frequently enough to have their own token, while rare combinations decompose into smaller pieces. Byte-Pair Encoding discovers these subwords empirically by greedily merging the most frequent character pairs until a target vocabulary size is reached.

**The mechanics:**

```
Start: character-level vocabulary
Repeat until vocab size reached:
    count all adjacent symbol pairs in corpus
    merge the most frequent pair into a new symbol
    update vocabulary and corpus

Example:
  "low lower lowest" → [l,o,w] [l,o,w,e,r] [l,o,w,e,s,t]
  Most frequent pair: (l,o) → "lo"
  → [lo,w] [lo,w,e,r] [lo,w,e,s,t]
  Next: (lo,w) → "low"
  → [low] [low,e,r] [low,e,s,t]
```

Practical tokenization rates:

| Input | Tokens | Reason |
|:---|:---|:---|
| "the" | 1 | Common word, own token |
| "unbelievable" | 3–4 | Split into subwords |
| "\`\`\`python" | 2 | Code-specific patterns |
| "2023-11-15" | 4–6 | Dates fragment |
| Chinese character | 1–3 | Non-Latin costly |

**What breaks:** BPE tokenization is language and domain-specific. A tokenizer trained on English web text gives fewer tokens per character to English than to Chinese or Arabic — multilingual models tokenize non-English text less efficiently, consuming more context for the same information. Numbers tokenize unpredictably: "100", "101", "102" may map to completely different tokens with no arithmetic relationship, contributing to LLM arithmetic failures. Rare technical terms fragment into many tokens, increasing inference cost and sometimes degrading model understanding.

---

## Attention Mechanism

**The problem:** at each position, the model must decide how much to weight information from every other position in the context. A fixed lookup table cannot do this — the relevance of position j to position i depends on the content at both positions, which changes with every input. A recurrent hidden state (RNN) forces all past information through a fixed-size bottleneck, making it hard to retrieve information from many steps back.

**The core insight:** for each query position, compute a content-based similarity score against all key positions, then use those scores as attention weights to aggregate value vectors. The model learns what to look for (query projection), what to advertise (key projection), and what to contribute (value projection). Content-based retrieval from arbitrary past positions, with gradients flowing directly back to every attended position.

**The mechanics:**

```
Attention(Q, K, V) = softmax(QKᵀ / √d_k) · V

where:
  Q = X·Wq  ← what this position is looking for
  K = X·Wk  ← what each position is advertising
  V = X·Wv  ← what each position contributes
  √d_k      ← prevents softmax saturation at high d
```

Causal mask: set attention scores at positions j > i to -∞ before softmax, ensuring each position attends only to itself and the past.

Multi-head attention runs h independent attention heads in parallel on projected subspaces, then concatenates:

```
MultiHead(Q,K,V) = Concat(head₁, ..., headₕ) · Wₒ
headᵢ = Attention(Q·Wᵢq, K·Wᵢk, V·Wᵢv)
```

Each head can specialize: one head may track coreference, another syntactic dependency, another local context.

**What breaks:** attention cost is O(N²) in sequence length — a 100K-token context requires 10B attention score computations per layer. For long contexts this is both slow and memory-intensive (the N×N attention matrix must be materialized unless FlashAttention is used). The softmax over many positions tends to attend strongly to a few tokens and weakly to most — attention is often sparse in practice, motivating sparse attention approximations.

---

## Positional Encoding

**The problem:** self-attention is permutation-equivariant — if you shuffle the input tokens, the output shuffles identically. The model has no built-in sense of order. "Dog bites man" and "Man bites dog" produce the same attention scores between "dog" and "man" unless position information is explicitly provided.

**The core insight:** inject position information into the token representations so that each position produces a distinct key and query. The encoding must be (1) unique per position, (2) accessible to the attention mechanism's dot products, and ideally (3) able to represent relative distances so the model can generalize to sequence lengths not seen in training.

**The mechanics — four approaches:**

Sinusoidal (original Transformer): fixed functions of position and dimension index, not learned:
```
PE(pos, 2i)   = sin(pos / 10000^{2i/d})
PE(pos, 2i+1) = cos(pos / 10000^{2i/d})
```

Learned absolute embeddings (GPT-2, GPT-3): an embedding table E[pos] trained end-to-end. Simple but cannot extrapolate to positions unseen during training.

RoPE (Rotary Position Embedding — LLaMA, Mistral): rotates Q and K vectors by an angle proportional to position before the dot product. The inner product between rotated Q at position m and K at position n depends only on their content and on (m - n), not on absolute positions:
```
RoPE(q, m) = q · e^{imθ}
```
Relative distance enters attention scores naturally. Length extrapolation by adjusting the base frequency (YaRN, LongRoPE).

ALiBi (Attention with Linear Biases — MPT, BLOOM): adds a distance penalty to attention scores directly, no encoding in Q or K:
```
score(i, j) = qᵢkⱼᵀ / √d_k − m·(i−j)
```
Different heads use different slopes m. Simple, no position embedding parameters, generalizes beyond training length.

**What breaks:** learned absolute embeddings are hard-limited to training length — position 4097 has no embedding if training saw only 4096. RoPE extrapolates better but degrades at lengths much larger than training (addressed by frequency interpolation). ALiBi's linear penalty may underweight distant but relevant tokens in very long documents.

---

## Feed-Forward Network (FFN) and SwiGLU

**The problem:** attention redistributes information across positions but does not transform the content at each position. After attention, token representations need nonlinear processing to compute complex functions of the attended information. A single linear layer is insufficient — linear functions cannot represent XOR-type interactions, and stacking linear layers collapses to one linear layer.

**The core insight:** a two-layer MLP with a nonlinear activation applied independently at each position is sufficient to approximate arbitrary functions. The FFN is wider than the model dimension (typically 4×) to give the network capacity to compute complex mappings before projecting back. The FFN is thought to store factual associations — neuron activations in the FFN correspond to knowledge retrieval.

**The mechanics:**

```
FFN(x) = W₂ · activation(W₁x + b₁) + b₂

For d_model = 4096:
  W₁ ∈ ℝ^{4096 × 16384}
  W₂ ∈ ℝ^{16384 × 4096}
```

SwiGLU (LLaMA, PaLM) — replaces standard ReLU/GELU:
```
SwiGLU(x) = Swish(W₁x) ⊙ (W₂x)
           = (W₁x · σ(W₁x)) ⊙ (W₂x)
```
Two separate projections are gated elementwise. Empirically outperforms ReLU and GELU at the same parameter count. Uses 3 weight matrices instead of 2; hidden dimension is reduced to 2/3 of 4× to keep parameter count equal.

**What breaks:** the FFN is the largest parameter consumer in a transformer — about 2/3 of total parameters in standard models. This drives both memory cost and compute cost. Widening the FFN is the primary way to scale model capacity, but it scales parameter count and FLOPS proportionally. MoE (next section) breaks this coupling.

---

## Mixture of Experts (MoE)

**The problem:** scaling an FFN increases parameters and compute proportionally. A 4× wider FFN requires 4× more matrix multiplications per forward pass. There is no way to add knowledge capacity (more parameters) without adding compute cost.

**The core insight:** replace one dense FFN with N expert FFNs but activate only k of them per token. Total parameters scale with N, but compute per token scales with k. The router learns to send tokens to the most relevant experts — a soft learned specialization.

**The mechanics:**
```
y = Σᵢ G(x)ᵢ · Eᵢ(x)

G(x) = TopK(softmax(Wg·x), k)
```

Each token selects k experts (typically k=2) with the highest router scores. Experts not in the TopK have zero weight — no compute for their FFN.

Examples: Mixtral 8×7B (8 experts, top-2 routing → 12.9B active parameters per token out of 46.7B total), DeepSeek-V2 (160 experts, top-6 routing).

Load balancing: a token that only routed to experts 1 and 2 would starve experts 3–8. An auxiliary load-balancing loss penalizes unequal utilization across experts during training.

**What breaks:** expert load imbalance causes under-utilization even with the auxiliary loss — some experts attract more tokens. In distributed inference, all experts must be loaded across devices even though only k/N are used per token; memory cost is N× a dense model even though compute is k/N×. The routing decision is discrete (TopK), making the gradient of the routing function zero almost everywhere — the auxiliary loss workarounds are heuristics.

---

## Pretraining

**The problem:** a randomly initialized transformer has no language knowledge, no world knowledge, and no ability to follow instructions. All knowledge and capability must be learned from data. The training objective determines what is learned.

**The core insight:** next-token prediction on a large corpus of human-generated text is sufficient to learn language, facts, reasoning patterns, and world knowledge — because producing accurate next-token predictions requires implicitly modeling all of these. This is the pretraining objective, applied to trillions of tokens from the internet, books, code, and papers.

**The mechanics:**
```
L = -Σₜ log P_θ(xₜ | x₁, ..., x_{t-1})
```

Standard autoregressive cross-entropy loss. During training, all positions compute their predictions in parallel (teacher forcing — the ground truth context is always provided, not the model's own predictions).

Chinchilla scaling laws: optimal allocation of a compute budget C between model size N and training tokens D:
```
N_opt ≈ √(C / 6)
D_opt ≈ √(6C)
```

Simplified rule: ~20 training tokens per parameter for a compute-optimal run. In practice, inference-optimized models over-train: LLaMA 2 7B was trained on 2T tokens (~285 tokens per parameter) to make the deployed model cheaper at inference even if training was not compute-optimal.

| Model | Parameters | Training tokens |
|:---|:---|:---|
| GPT-3 | 175B | 300B (under-trained by Chinchilla) |
| LLaMA 2 7B | 7B | 2T (over-trained for inference) |
| LLaMA 2 70B | 70B | 2T |

**What breaks:** pretraining produces a capable but unaligned model — it predicts next tokens in the style of whatever it was trained on, including low-quality, harmful, and inconsistent text. It follows the form of internet text, not instructions. Instruction following and alignment require additional training stages (SFT, RLHF, DPO).

---

## Instruction Tuning and Alignment

**The problem:** a pretrained model generates text that "sounds like" its training corpus. Given a question, it is as likely to generate another question (as on a Q&A site) as to generate an answer. It does not understand that the user wants a helpful response rather than a continuation of the document.

**The core insight:** show the model examples of the desired input-output behavior — (system prompt, user message, correct assistant response) triplets — and train it with the same next-token objective but on these demonstrations only. The model learns to produce the format it was shown. Alignment then goes further: use human preference data to reward helpful and safe responses over unhelpful or harmful ones.

**The mechanics — three-stage pipeline:**

Stage 1 — Pretraining: broad capabilities from massive web-scale corpus.

Stage 2 — SFT: fine-tune on high-quality demonstrations:
```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Summarize this document."},
    {"role": "assistant", "content": "The document describes..."}
  ]
}
```

Stage 3 — RLHF or DPO: optimize for human preference. RLHF trains a reward model on ranked responses, then uses PPO to maximize reward. DPO directly optimizes preference pairs without a separate reward model.

**What breaks:** SFT teaches format and task execution but not truthfulness — the model mimics demonstrations, including errors. RLHF fixes alignment but introduces reward hacking and requires expensive human annotations. The three-stage pipeline assumes each stage is applied to the same model; skipping or reordering stages produces unpredictable results.

---

## KV Cache

**The problem:** autoregressive generation is sequential — each new token requires a forward pass through all L layers. At each step, the attention mechanism must compute keys and values for all preceding tokens. Without caching, a 1000-token generation requires recomputing 500,000 key-value pairs (1000 steps × 500 average past tokens) instead of 1000.

**The core insight:** keys and values for past tokens do not change — once computed, they can be cached and reused. Each generation step only requires computing the new token's query, key, and value, then appending the new KV pair to the cache and attending over the full cached history.

**The mechanics:**
```
Step t:
  compute K_t, V_t for new token
  K_cache ← [K_cache; K_t]    ← append
  V_cache ← [V_cache; V_t]    ← append
  attention output ← softmax(Q_t · K_cache^T / √d_k) · V_cache
```

Memory footprint per token in the cache:
```
2 × L × H × d_h × bytes_per_element

For LLaMA-2 70B (80 layers, 64 heads, 128 head dim) in bfloat16:
= 2 × 80 × 64 × 128 × 2 bytes = 2.62 MB per token
At 4096 tokens: ~10.7 GB just for KV cache
```

Grouped-Query Attention (GQA) reduces this by sharing K and V heads across multiple Q heads — 8 KV heads for 64 Q heads reduces KV cache 8× with minimal quality loss.

**What breaks:** KV cache memory scales linearly with sequence length and batch size. At long contexts (128K+ tokens) or large batches, KV cache memory dominates and limits throughput. PagedAttention (vLLM) solves fragmentation — KV cache for different sequences is stored in non-contiguous pages, like OS virtual memory, eliminating 20–40% memory waste from static allocation.

---

## FlashAttention

**The problem:** standard attention materializes the N×N attention score matrix in GPU high-bandwidth memory (HBM). For N=8192, this is 8192² × 2 bytes = 134MB per layer per head — and the matrix must be written to HBM and read back for the softmax and the final matrix multiply. At long contexts, this memory traffic is the bottleneck, not compute.

**The core insight:** the attention output does not require materializing the full N×N matrix. By tiling the computation into blocks that fit in SRAM (on-chip fast memory), and fusing the softmax and weighted sum into a single pass through the data, the N×N matrix is never written to HBM. The same result is computed with O(N) HBM memory instead of O(N²), and significantly fewer HBM reads/writes.

**The mechanics:**
```
Tile Q into blocks of size Br
Tile K, V into blocks of size Bc
For each block of Q:
    Load from HBM to SRAM
    For each block of K, V:
        Load from HBM to SRAM
        Compute attention scores for this block
        Update running softmax (online algorithm)
        Accumulate weighted V contribution
Write final output to HBM
```

The online softmax algorithm maintains running max and denominator to compute the exact softmax without seeing all scores at once. FlashAttention 2 improves parallelism; FlashAttention 3 uses NVIDIA Hopper's tensor memory accelerator for further gains. Both are mathematically exact — not approximate.

**What breaks:** FlashAttention requires custom CUDA kernels — not portable to hardware without CUDA support. Tiling requires the block sizes to be chosen to fit SRAM, which is hardware-specific. The training backward pass is more complex because the N×N matrix was not stored; gradients require recomputation of attention scores from saved Q, K, V.

---

## Autoregressive Inference and Speculative Decoding

**The problem:** generating each token requires one full forward pass through all N layers of the target model. On a 70B model in bfloat16 (140GB of weights), each forward pass reads 140GB from HBM. At ~2TB/s HBM bandwidth, this sets a hard floor of ~70ms per token. The GPU's tensor cores are largely idle — the bottleneck is memory bandwidth, not compute.

**The core insight:** a small, fast draft model can propose K tokens cheaply. The large target model can verify all K proposals in a single parallel forward pass — the same cost as generating one token — via rejection sampling. If most proposals are accepted, K tokens are produced for approximately the cost of one target model pass.

**The mechanics:**
```
For each generation step:
1. Draft: run small model autoregressively for K steps → tokens x̃₁...x̃ₖ with probs p₁...pₖ
2. Verify: run target model once on full input + K draft tokens → probs q₁...qₖ
3. Accept/reject left to right:
   For token i:
     with prob min(1, qᵢ(x̃ᵢ)/pᵢ(x̃ᵢ)): ACCEPT x̃ᵢ
     else: REJECT, sample from max(0, q-p)/Z, STOP
4. Output all accepted tokens + one bonus token
```

Expected tokens per target model call at acceptance rate α with K draft tokens:
```
E[tokens] = (1 - α^{K+1}) / (1 - α)

At α=0.8, K=4: E[tokens] ≈ 3.36
At α=0.9, K=4: E[tokens] ≈ 4.10
```

Output distribution is identical to the target model — rejection sampling guarantees this.

**What breaks:** speedup depends on acceptance rate. High-temperature sampling widens the gap between draft and target distributions, reducing acceptance. Large batch sizes eliminate the memory-bandwidth bottleneck (GPU is already compute-bound), making draft overhead net-negative. Requires draft and target to share vocabulary.

---

## Context Window and Long Context

**The problem:** the KV cache grows linearly with context length; attention cost grows quadratically; and positional encodings are trained at a fixed maximum length. Each of these independently limits how much context a model can process.

**The core insight:** each bottleneck requires its own solution. Memory: Grouped-Query Attention and PagedAttention reduce KV cache cost. Compute: FlashAttention reduces the quadratic constant. Position: RoPE frequency interpolation (YaRN, LongRoPE) extends the effective context range by rescaling the positional frequencies the model was trained with.

**Context sizes by model:**

| Model | Context window |
|:---|:---|
| GPT-3 | 4,096 tokens |
| GPT-4 Turbo | 128,000 tokens |
| Claude 3.5 Sonnet | 200,000 tokens |
| Gemini 1.5 Pro | 1,000,000 tokens |

YaRN (Yet Another RoPE extensioN): splits RoPE dimensions into low-frequency (interpolate) and high-frequency (extrapolate) groups, applying different scaling strategies to each. Enables 4–32× context extension with minimal fine-tuning.

Ring Attention: distributes the attention computation across devices by passing KV blocks in a ring topology. Each device computes attention for its local Q slice against the full K, V sequence — enabling contexts of millions of tokens across a GPU cluster.

**What breaks:** "lost in the middle" — LLMs recall information at the start and end of long contexts more reliably than the middle. The U-shaped recall pattern persists even for models with 1M-token contexts. Adding more context window capacity does not fix the model's ability to actually use it.

---

## Inference Architectures

**The problem:** serving an LLM at scale requires high throughput (many requests per second) and low latency (fast response per user). These objectives conflict: batching improves throughput but adds latency. Memory constraints limit batch size. Different requests have different lengths, making static allocation wasteful.

**The core insight:** the serving bottleneck is the KV cache, not compute. Systems that manage KV cache memory efficiently enable larger effective batch sizes and higher throughput. Continuous batching (replacing finished sequences with new ones mid-batch rather than waiting for all to finish) and PagedAttention (OS-style virtual memory for KV pages) together eliminate the two largest sources of inefficiency.

| Approach | How | Latency | Throughput |
|:---|:---|:---|:---|
| Single GPU, static batching | Pad all requests to max length | Low | Low |
| Tensor parallelism | Split weight matrices across GPUs | Medium | High |
| Pipeline parallelism | Split layers across GPUs | High (bubbles) | High |
| Continuous batching | Replace finished requests mid-batch | Low per-token | High |
| PagedAttention (vLLM) | OS-style KV cache pages | Low | Very high |

PagedAttention: KV cache is divided into fixed-size pages (blocks of ~16 tokens). Each sequence gets a logical page table mapping to physical blocks. Non-contiguous physical memory is used, eliminating internal fragmentation. Enables 24× higher throughput than naive HuggingFace Transformers at the same memory.

**What breaks:** tensor parallelism requires all-reduce operations across GPUs on every forward pass — communication overhead scales with model parallelism degree and becomes a bottleneck on slow interconnects (PCIe vs. NVLink). Pipeline parallelism creates pipeline bubbles — GPUs in early stages are idle while late stages process. Continuous batching requires variable-length attention kernels and careful scheduler design to balance latency fairness across requests.

---

## Key Metrics

**The problem:** "the model is fast" is not a testable claim. Serving quality requires decomposing latency into its components, because different components respond to different optimizations.

**The core insight:** TTFT (time to first token) and TPOT (time per output token) measure different things. TTFT is dominated by the prefill pass — compute cost scales with input length. TPOT is dominated by memory bandwidth — the cost of loading weights for each decode step. Optimizing one does not necessarily optimize the other.

| Metric | Definition | Typical value | Optimized by |
|:---|:---|:---|:---|
| TTFT | Time from request to first output token | 100ms–2s | Smaller model, batched prefill, faster hardware |
| TPOT | Time per output token after first | 20–100ms | Speculative decoding, quantization, higher memory bandwidth |
| Throughput | Output tokens per second across all requests | Hardware-dependent | Continuous batching, PagedAttention, tensor parallelism |
| Perplexity | exp(−(1/N)Σ log P(wᵢ|w<ᵢ)) | Lower = better LM | More data, larger model, better architecture |

Perplexity measures language model quality on held-out text; it correlates with downstream task performance but is not a substitute for task-specific benchmarks.

**What breaks:** optimizing TPOT with speculative decoding does not help TTFT. Optimizing throughput with large batch sizes increases per-user latency. These tradeoffs are inherent — production serving requires explicit decisions about which metric the system is optimized for, based on the application's requirements.

*Related: [Inference Optimization](inference-optimization.md) | [Tuning and Optimization](tuning-optimization.md) | [Speculative Decoding](speculative-decoding.md)*
