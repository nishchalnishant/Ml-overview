# State Space Models

How Mamba, RWKV, and Jamba solve the O(n²) quadratic attention bottleneck — with the mathematical intuition behind selective state spaces, the recurrence-convolution duality, and the engineering trade-offs that determine when to use SSMs over transformers.

---

## 1. Core Concept & Intuition

The quadratic attention bottleneck is concrete: for a sequence of length n, computing attention requires O(n²) operations and O(n²) memory. At n=1K tokens: 1M operations. At n=1M tokens: 1T operations — 1 billion× more expensive. Long-context applications (genomics, audio, entire codebases) make this cost prohibitive.

**What older paradigms couldn't do:**

Standard RNNs (LSTM, GRU) are O(n) but suffer from the vanishing gradient problem — gradients decay exponentially over long sequences because the hidden state update is a matrix multiplication that contracts the gradient norm. After ~1,000 timesteps, the gradient from early tokens is numerically zero.

Transformers eliminate the vanishing gradient (direct attention from any token to any other token — gradient path length = 1 regardless of sequence length) but pay the O(n²) cost.

**The SSM insight:** Model sequences via a linear dynamical system that is: (1) efficiently computable as a recurrence for inference (O(1) per step, O(n) total), and (2) parallelizable as a convolution during training (O(n log n) via FFT). The key innovation in Mamba is making this system **selective** — the state transition matrices change per-input, giving the model the ability to choose what to remember.

```
Core tension:
  Fixed dynamics (classic SSM/RNN): O(n) compute, O(1) per step — BUT fixed dynamics
                                     cannot selectively forget or attend
  Selective dynamics (Mamba):        Same O(n) compute — dynamics change per input token
                                     Enables selective attention-like behavior
  Transformers:                       O(n²) compute — arbitrary attention, no compression
```

---

## 2. Architecture & Mathematics

### 2.1 The Continuous State Space Model

Classical SSMs come from control theory. A continuous-time linear dynamical system:

```
Continuous-time SSM:
  x'(t) = A·x(t) + B·u(t)    [state update]
  y(t)  = C·x(t) + D·u(t)    [output]

where:
  u(t) ∈ ℝ^d_in   — input signal at time t
  x(t) ∈ ℝ^N      — hidden state (N is the state dimension, typically 16-64)
  y(t) ∈ ℝ^d_out  — output signal
  A ∈ ℝ^{N×N}     — state transition matrix (how state evolves)
  B ∈ ℝ^{N×d_in}  — input projection (how input affects state)
  C ∈ ℝ^{d_out×N} — output projection (how state produces output)
  D ∈ ℝ^{d_out×d_in} — skip connection (usually small or zero)
```

For discrete sequences (text), we discretize using the Zero-Order Hold method with step size Δ:

```
Discretization (ZOH):
  Ā = exp(Δ·A)                    [matrix exponential]
  B̄ = (Δ·A)^{-1}(exp(Δ·A) - I)·Δ·B   [integral of continuous input]
  
  Simplification (for diagonal A):
  Ā = exp(Δ·A)   (element-wise if A is diagonal)
  B̄ = (exp(Δ·A) - 1) / A · B  (element-wise)
  C̄ = C, D̄ = D
```

Discrete recurrence:
```
x_k = Ā·x_{k-1} + B̄·u_k
y_k = C̄·x_k + D̄·u_k
```

This is a linear recurrence — O(1) per step, O(n) total for a sequence of length n.

### 2.2 The Recurrence-Convolution Duality

The linear recurrence can be unrolled into a convolution — this is the key to making SSMs parallelizable:

```
Unrolling the recurrence:
  x_1 = B̄·u_1
  x_2 = Ā·x_1 + B̄·u_2 = Ā·B̄·u_1 + B̄·u_2
  x_3 = Ā·x_2 + B̄·u_3 = Ā²·B̄·u_1 + Ā·B̄·u_2 + B̄·u_3
  
  y_k = C̄·x_k = Σ_{j=0}^{k-1} C̄·Āʲ·B̄·u_{k-j}

This is a convolution:
  y = u * K   where K_j = C̄·Āʲ·B̄  (the SSM kernel)
  K = [C̄·B̄, C̄·Ā·B̄, C̄·Ā²·B̄, ..., C̄·Ā^{n-1}·B̄]
```

**During training:** precompute the kernel K for the entire sequence length, then apply as a convolution via FFT: O(n log n). All positions compute in parallel — no sequential dependence.

**During inference:** use the recurrence form — maintain hidden state x_k, update with new token: O(1) per new token, O(n) total. No need to recompute all previous attention keys/values.

```
Training mode (parallel convolution):   O(n log n) time, O(n) memory
Inference mode (sequential recurrence): O(1) per step, O(N) memory (just the state!)

Compare to Transformer:
Training:  O(n²) time, O(n²) memory
Inference: O(n) per step (KV cache grows), O(n) memory
```

The SSM has no "KV cache" — the entire history is compressed into a fixed-size state vector of dimension N. This is both the strength (constant memory at inference) and the weakness (lossy compression — some information is inevitably lost).

### 2.3 HiPPO: The Key to Stable Long-Range Memory

Naively choosing A randomly leads to vanishing gradients (eigenvalues of A less than 1 → state decays). HiPPO (High-order Polynomial Projection Operator, Gu et al., 2020) provides a theoretically grounded initialization for A:

```
HiPPO-LegS matrix (Legendre Polynomials, shifted):
  A_{nk} = -(2n+1)^{1/2} · (2k+1)^{1/2}  if n > k
           = -(n+1)                         if n = k
           = 0                              if n < k

This A matrix maintains polynomial projections of the input history —
it provably preserves information about the entire past sequence
with decreasing precision for more distant history.
```

The HiPPO matrix is a structured initialization that ensures A has eigenvalues on or near the unit circle — the state neither explodes nor vanishes. All modern SSMs (S4, Mamba) use variants of this structured A.

### 2.4 S4: Structured State Space Models

S4 (Gu et al., 2022) made SSMs practically competitive with transformers by:

1. **Diagonal-plus-low-rank (DPLR) structure for A:**
   ```
   A = Λ - P·Qᵀ   (Λ diagonal, P and Q low-rank)
   ```
   This structure allows the matrix exponential exp(Δ·A) to be computed efficiently using the Cauchy kernel formula — reducing O(N²) matrix operations to O(N) operations.

2. **Parallel kernel computation:**
   The SSM kernel K can be computed in O(N + L) time for sequence length L using the DPLR structure, making training O(L log L) via FFT convolution.

### 2.5 Mamba: Selective State Spaces

S4 and prior SSMs have a critical limitation: A, B, C, Δ are **time-invariant** (the same for every token in the sequence). The model cannot "pay attention" to some tokens and ignore others — it processes every token with the same dynamics.

**Mamba's selective mechanism:** make B, C, Δ **input-dependent** (change per token):

```
Standard SSM (S4):
  Ā = f(Δ, A)         ← fixed: same Δ for all tokens
  B̄ = f(Δ, A, B)      ← fixed: same B for all tokens
  
  x_k = Ā·x_{k-1} + B̄·u_k   ← dynamics don't depend on what u_k is

Mamba (Selective SSM):
  Δ_k = softplus(Linear(u_k))    ← token-dependent Δ!
  B_k = Linear(u_k)              ← token-dependent B!
  C_k = Linear(u_k)              ← token-dependent C!
  
  Ā_k = exp(Δ_k ⊙ A)            ← per-token state transition
  B̄_k = (exp(Δ_k ⊙ A) - 1) / A ⊙ B_k
  
  x_k = Ā_k ⊙ x_{k-1} + B̄_k ⊙ u_k   ← selective: can gate what enters state
  y_k = C_k · x_k
```

**Intuition for selectivity:**

Large Δ_k: `Ā_k = exp(Δ·A)` → large exponent → the previous state x_{k-1} is "forgotten" more completely; the current input u_k dominates. The model "focuses" on the current token.

Small Δ_k: Ā_k ≈ I; the state is carried forward with minimal change; the current input has little effect. The model "ignores" the current token and remembers the past.

This is analogous to attention scores in transformers: high attention = focus on this token; low attention = this token doesn't affect the representation.

**Why selectivity breaks the convolution trick:**

The convolution trick requires time-invariant kernels. With Ā_k, B̄_k changing per token, the kernel K changes at every position — you can't precompute it. Mamba can't use FFT convolution during training.

**Mamba's solution — hardware-aware parallel scan:**

A parallel prefix scan (parallel sum/product) algorithm can compute all x_k = Ā_k ⊙ x_{k-1} + B̄_k ⊙ u_k simultaneously:

```
Parallel scan for a[0], a[1], ..., a[n-1] (outputs cumulative products/sums):
  Round 1: compute local pairs (a[0]⊗a[1]), (a[2]⊗a[3]), ... — n/2 pairs in parallel
  Round 2: compute (a[0]⊗...⊗a[3]), (a[4]⊗...⊗a[7]), ...  — n/4 pairs in parallel
  ...
  Round log(n): one final merge
  
Total work: O(n log n), but O(log n) parallel steps — O(n log n) time with parallelism → O(n) actual wall time
```

Mamba implements this as a custom CUDA kernel that keeps the scan entirely in SRAM (no HBM reads), achieving near-peak hardware utilization.

**Mamba block architecture:**

```
Input: u ∈ ℝ^{B × L × d_model}

x = Linear(u)              # expand: d_model → d_expand (typically 2×)
z = Linear(u)              # gate branch: d_model → d_expand

x = Conv1d(x)              # local context: causal 1D conv, kernel size 4
x = SiLU(x)

# SSM with selective parameters:
Δ = softplus(Linear(x))    # [B, L, d_expand] per-token step size
B = Linear(x)              # [B, L, d_state]
C = Linear(x)              # [B, L, d_state]
y = SSM(x, Δ, A, B, C)    # selective state space scan

y = y ⊙ SiLU(z)           # output gate
output = Linear(y)         # project back: d_expand → d_model
```

The output gating `y ⊙ SiLU(z)` (from Gated SSM) allows the model to selectively zero out SSM outputs that aren't useful — analogous to the gate in LSTM.

### 2.6 RWKV: Linear Attention via Decay

RWKV (Peng et al., 2023) takes a different approach: reformulate attention to be computable as a linear recurrence by replacing the softmax with a weighted decay.

**Standard attention (for reference):**
```
Attention(Q, K, V) = softmax(QKᵀ/√d) · V
```

The softmax makes this O(n²) — you must compute all pairwise QK scores.

**RWKV's linear recurrence formulation:**

For token at position t, RWKV computes output as:

```
o_t = (Σ_{i≤t} exp(-(t-i)·w + k_i) · v_i) / (Σ_{i≤t} exp(-(t-i)·w + k_i))

where w = per-channel decay (learned, positive) — attention decays exponentially with distance
      k_i, v_i = key/value at position i

This is a weighted average of values, where weights decay exponentially with distance:
  weight at position i (from position t) ∝ exp(-(t-i)·w) · exp(k_i)
                                         = exp(k_i) · γ^{t-i}  where γ = exp(-w) ∈ (0,1)
```

This can be computed as a recurrence (a_t, b_t updated one step at a time):

```
a_t = exp(-w) · a_{t-1} + exp(k_t) · v_t   [numerator recurrence]
b_t = exp(-w) · b_{t-1} + exp(k_t)          [denominator recurrence]
o_t = a_t / b_t
```

O(1) per step. No KV cache needed. The entire history is compressed into a_t, b_t (both size d_head).

**RWKV vs Mamba:**

| Aspect | RWKV | Mamba |
|---|---|---|
| Memory mechanism | Exponential decay (fixed rate) | Selective state space (learned, per-token) |
| Selectivity | Limited (w is position-independent) | Full (Δ, B, C all input-dependent) |
| Training | Fully recurrent (or custom "time-mixing" attention) | Parallel scan CUDA kernel |
| Architecture heritage | Transformer-like (still has Q, K, V structure) | Control-theory SSM |
| RWKV-v6 quality | ~90% of transformer on standard benchmarks | ~95% of transformer at same parameter count |

RWKV's main advantage: simpler implementation than Mamba's custom CUDA kernel. RWKV "time-mixing" is compatible with standard deep learning frameworks.

### 2.7 Jamba: Hybrid Transformer-Mamba

Jamba (AI21 Labs, 2024) interleaves transformer attention layers and Mamba SSM layers:

```
Jamba architecture (ratio: 1 attention : 7 Mamba per block):
  Block 1: [Mamba, Mamba, Mamba, Mamba, Mamba, Mamba, Mamba, Attention]
  Block 2: [Mamba, Mamba, Mamba, Mamba, Mamba, Mamba, Mamba, Attention]
  ...
  
Also uses MoE FFN layers (Mixtral-style) in some layers.
```

**Rationale:** Mamba handles long-range sequential dependencies efficiently. Attention layers provide the "looking back at anything" capability that is hard for SSMs (which compress history lossy). 1 attention layer per 7 Mamba layers provides sufficient global attention capability at a fraction of the quadratic cost.

**Jamba's context capacity:** 256K tokens at practical throughput — the 7:1 Mamba:attention ratio reduces the quadratic attention cost by 7×. The Mamba layers handle most of the long-range information; attention layers provide precise long-range recall for critical positions.

**KV cache behavior:**
```
Pure Transformer at 256K context:
  KV cache = 2 × L × H × d_head × n_tokens × 2 bytes
  Very large — may exceed GPU memory

Jamba at 256K context:
  KV cache = 2 × (L/8) × H × d_head × n_tokens × 2 bytes  (only attention layers)
  Mamba state = N × d_expand × L × 2 bytes  (fixed-size, independent of context length)
  
  ~7× KV cache reduction vs pure transformer, enabling 256K on a single A100
```

---

## 3. Trade-offs & System Design Implications

### Fundamental Trade-off Table

| Property | Transformer | Mamba (SSM) | Hybrid (Jamba) |
|---|---|---|---|
| Training compute | O(n²) | O(n log n) | O(n² × 1/8 + n log n × 7/8) |
| Inference per step | O(n) — KV cache grows | O(1) — fixed state | O(n × 1/8) |
| Memory at inference | O(n) — grows with context | O(N) — fixed (N=state dim) | O(n × 1/8 + N) |
| In-context recall | Perfect (any token, any distance) | Lossy (info compressed into state) | Good (attention layers handle recall) |
| Training parallelism | Full | Requires custom parallel scan | Full |
| Throughput at long context | Degrades quadratically | Constant | Sublinear degradation |

### When SSM/Hybrid Outperforms Transformer

**SSMs are preferable when:**
- Sequences are very long (>100K tokens): genomics (entire chromosomes), audio (waveforms at 16kHz → millions of samples), time series, video frame sequences
- Deployment is memory-constrained: the fixed-size SSM state is constant regardless of sequence length — no KV cache memory growth
- Streaming inference: new tokens processed O(1) without growing memory; transformers require O(n) computation and O(n) memory per new token with KV cache

**Transformers remain preferable when:**
- Task requires precise retrieval of specific past tokens: "in the document at page 2, what was the exact definition of X?" — SSMs cannot reliably retrieve specific tokens because history is lossy
- Training data is large and training compute is not the bottleneck: at the same parameter count, transformers currently achieve slightly higher quality on most benchmarks
- Short sequences (<10K tokens): the O(n²) cost is manageable; transformer's advantages (better in-context learning, precise recall) dominate

### SSM Quality Gap

Current state (2025): Mamba-2 and Jamba achieve 90-95% of equivalent-size transformer quality on standard NLP benchmarks. The remaining 5-10% gap is most pronounced on:
- Needle-in-a-haystack tasks (find specific information in very long context) — SSMs compress and may lose the needle
- In-context learning with many examples — transformers leverage all examples exactly; SSMs may underweight early examples
- Associative recall ("what is the value paired with key X?") — SSMs fail when X appeared only once early in a long sequence

The hybrid approach (Jamba) largely closes this gap: attention layers provide exact recall; Mamba layers handle the bulk of sequence processing efficiently.

---

## 4. Canonical Interview Q&As

**Q1: Explain the mathematical connection between the recurrence and convolution forms of an SSM. Why does selectivity (Mamba) break the convolution trick, and how does Mamba compensate?**

The recurrence `x_k = Ā·x_{k-1} + B̄·u_k` can be unrolled:
`y_k = C̄·x_k = Σ_{j=0}^{k-1} C̄·Āʲ·B̄·u_{k-j}`

This is a linear convolution y = u * K where the SSM kernel is `K_j = C̄·Āʲ·B̄`. When Ā, B̄, C̄ are time-invariant (don't change with the input), K is a fixed kernel and can be computed once and applied via FFT in O(n log n) — this is the standard SSM training approach.

Mamba makes B, C, Δ (and thus Ā) input-dependent: Ā_k = exp(Δ_k ⊙ A) where Δ_k = softplus(Linear(u_k)). Now the "kernel" changes at every position — there's no single K to precompute; the kernel at position t depends on all inputs u_1, ..., u_t. FFT convolution requires a time-invariant kernel — it cannot be applied.

Mamba compensates with a hardware-aware parallel scan (also called the parallel prefix scan or associative scan). The recurrence `x_k = Ā_k ⊙ x_{k-1} + B̄_k ⊙ u_k` is a linear recurrence with time-varying coefficients. It can be computed in O(n log n) parallel steps using the parallel prefix scan algorithm: in round i, each element combines with its neighbor 2^{i-1} positions ahead. After log(n) rounds, all positions have their correct x_k. Mamba implements this as a fused CUDA kernel that keeps all intermediate values in fast on-chip SRAM (shared memory), avoiding the slow HBM reads that would naively be required.

**Q2: What is the HiPPO matrix and why is it necessary for SSMs to model long-range dependencies?**

Without careful initialization, the state matrix A in a linear SSM either causes the hidden state to vanish (eigenvalues < 1 → exponential decay) or explode (eigenvalues > 1). Vanishing means the model forgets distant history; exploding makes training unstable.

The HiPPO (High-order Polynomial Projection Operator) matrix is derived from the theory of function approximation. The goal: define a matrix A such that the hidden state x(t) always contains an optimal polynomial approximation of the input history u(τ) for τ ≤ t. "Optimal" means the state minimizes the expected squared approximation error under a specific measure.

For the Legendre shifted measure (uniform over [0, t]):
```
A_{nk} = -(2n+1)^{1/2}(2k+1)^{1/2}  if n > k
         = -(n+1)                      if n = k
         = 0                           if n < k
```

The eigenvalues of this matrix lie exactly on the imaginary axis (purely imaginary) — no decay, no growth, perfect stability. The hidden state geometrically represents the coefficients of the degree-N polynomial approximation of the past N steps. The lower-degree polynomial coefficients capture slow trends (long-range); higher-degree coefficients capture rapid variations (short-range).

In practice: S4 and Mamba initialize A using the HiPPO matrix and keep A frozen or constrained to DPLR structure (diagonal-plus-low-rank), which preserves the favorable eigenvalue structure while allowing efficient computation of exp(Δ·A).

**Q3: Compare Mamba and a Transformer with sliding window attention on the task of "retrieve the exact string from 100K tokens ago." Which succeeds, why does the other fail, and what architectural modification would fix the failure?**

**Transformer with sliding window attention (e.g., window W=4K):** The query at position t can only attend to positions t-W to t. A token at position t-100K is outside the window and cannot be directly attended to. The information must propagate through ~25 layers of 4K-window attention blocks — but with residual connections and the right token representations, some information may leak through. In practice, exact retrieval of a specific string from 100K positions ago fails with sliding window attention — the token is simply not in the attention window at any layer.

**Fix:** Periodically insert full-attention layers (like Jamba's 1:7 ratio, or Longformer's global tokens) that can attend to any position. Or use ALiBi/RoPE with appropriate decay to allow longer windows at some layers.

**Mamba:** At position t, the hidden state x_t is a compressed summary of all prior inputs u_1, ..., u_{t-1}. Whether a specific string from t-100K is recoverable depends on Mamba's selectivity. If the string appeared once briefly and was "overwritten" by subsequent inputs (Ā_k >> 0 for many subsequent tokens → state updated repeatedly), the exact string may be lost from x_t. Mamba's selective mechanism can in principle learn to "freeze" the state when important tokens appear (by generating small Δ, keeping Ā_k ≈ I and preventing future overwriting) — but this requires the model to recognize importance at the time of encoding, before knowing the query.

**Empirical outcome:** On needle-in-a-haystack benchmarks, pure Mamba models degrade significantly beyond 50-100K tokens for precise string retrieval. Transformers with full attention succeed but at quadratic cost. Jamba (hybrid) succeeds because the attention layers can attend precisely to the needle token's position, while Mamba layers handle the bulk efficiently.

**Fix for Mamba:** External memory augmentation — when a critical token is processed, explicitly write it to an external key-value store. At retrieval time, query the store. This transforms Mamba from a purely compressive model to one with perfect recall for explicitly indexed tokens, at the cost of memory overhead.

**Q4: Derive the time and memory complexity of Mamba during training vs inference. Compare to a Transformer at sequence length n=1M.**

**Mamba training (parallel scan):**
```
Time: O(n log n) for the parallel scan + O(n) for linear projections → O(n log n)
Memory: O(n) to store inputs, intermediate scan states, and outputs
        (no n² attention matrix)

At n=1M:
  Time: 1M × log₂(1M) = 1M × 20 = 20M operations per Mamba layer
  Memory: O(1M) = 1M × d_expand × 2 bytes ≈ 2GB for d_expand=256 and 4 bytes
```

**Transformer training (standard attention):**
```
Time: O(n²) for attention matrix computation
Memory: O(n²) for the attention matrix

At n=1M:
  Time: 1M² = 10¹² = 1 trillion operations per attention layer
  Memory: 1M² × 2 bytes = 2 × 10¹² bytes = 2 TB — completely infeasible
```

**Mamba inference (per new token):**
```
Time: O(1) — just update x_k = Ā_k ⊙ x_{k-1} + B̄_k ⊙ u_k, compute y_k = C_k · x_k
Memory: O(N) — just the hidden state x ∈ ℝ^{d_expand × N}
        For d_expand=256, N=16: 256 × 16 × 2 bytes = 8KB per layer — essentially zero

At n=1M tokens processed so far, adding token 1M+1:
  Mamba: O(1) compute, O(8KB × n_layers) memory
  Transformer: O(n) = O(1M) compute (must run over entire KV cache), 
               O(n × d × 2 bytes) = O(1M × 256 × 2) = 500MB KV cache memory
```

**Practical implication:** For streaming applications that maintain long conversational context, Mamba's inference memory stays constant regardless of conversation length. A 90-minute voice conversation (at 16kHz, 10 tokens/second ≈ 54K tokens) would require 54K × 327KB ≈ 17GB KV cache in a Llama-3-70B transformer. Mamba's state remains at 8KB × 80 layers = 640KB — three orders of magnitude smaller.

**Q5: Why do hybrid models like Jamba achieve near-transformer quality while retaining SSM efficiency? What is the minimum fraction of attention layers needed for quality parity?**

Hybrid models succeed because Mamba and attention are complementary in their failure modes:

**Mamba's failure mode:** Cannot reliably retrieve specific tokens from long history (lossy compression). Cannot implement "soft lookup" over arbitrary past positions. Tends to underperform on tasks that require attending to specific positions in a structured way.

**Attention's failure mode:** O(n²) computation makes long contexts intractable. No fixed-size memory — KV cache grows linearly.

In Jamba's 1:7 ratio (one attention layer per 7 Mamba layers), the attention layers handle the tasks requiring precise recall while Mamba handles the bulk of sequential processing. The hypothesis: most language modeling does not require arbitrary long-range recall; local context and slow-changing state are sufficient for most tokens. Only specific tasks (answering questions from earlier in the document, maintaining exact references) require attention-style recall. The sparse attention layers handle these cases.

**Minimum attention fraction:** Empirical results (Jamba paper, Zamba, various ablations) suggest:

```
Attention fraction → Quality (vs pure transformer, at fixed params):
  0% (pure Mamba):   ~90-93% on standard NLP benchmarks, much lower on retrieval tasks
  12.5% (1:7 ratio): ~97-98% on standard benchmarks, near-full retrieval quality
  25% (1:3 ratio):   ~99%, essentially indistinguishable from transformer
  50% (1:1 ratio):   100% (you've essentially built a transformer with SSM FFNs)
```

The 12.5% (1:7) threshold is practically significant: it reduces quadratic attention cost by 7× while recovering most of the quality gap vs pure Mamba. Below 10%, recall-heavy tasks degrade noticeably. The right ratio depends on the intended use case: for pure text generation where users don't ask specific recall questions, 5% attention may suffice; for RAG or document QA, 12-25% is needed.

Theoretically: information theory suggests that a fixed-size Mamba state cannot store more than O(N·log(vocab_size)) bits of information. For N=16 state dims and vocab 50K: ~800 bits per layer — sufficient for slow-changing contextual information but insufficient for precise storage of arbitrary token identities. Attention layers provide O(d·n) bits of capacity (the KV cache) — perfect retrieval at O(n) memory. Hybrid models distribute these two memory regimes: SSM for dense, slowly-varying context; attention KV cache for precise sparse facts.
