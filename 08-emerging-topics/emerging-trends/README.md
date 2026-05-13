# Emerging Trends in ML (2023–2025)
> Interview prep reference. Covers architectures, paradigms, and open problems that are actively shaping the field right now.

---

## Table of Contents
1. [State Space Models (SSMs)](#1-state-space-models-ssms)
2. [Test-Time Compute Scaling](#2-test-time-compute-scaling)
3. [Mixture of Experts (MoE)](#3-mixture-of-experts-moe)
4. [Long Context & Efficient Attention](#4-long-context--efficient-attention)
5. [Synthetic Data & Data Flywheels](#5-synthetic-data--data-flywheels)
6. [Multimodal Foundation Models](#6-multimodal-foundation-models)
7. [Efficient Fine-tuning Evolution](#7-efficient-fine-tuning-evolution)
8. [AI Agents & MCP](#8-ai-agents--mcp)
9. [AI Safety Frontiers](#9-ai-safety-frontiers)
10. [Common Interview Questions](#10-common-interview-questions)

---

## 1. State Space Models (SSMs)

### Why Transformers Hit a Wall

Transformers are the backbone of modern LLMs, but they carry a fundamental computational burden. Self-attention is **O(n²)** in both time and memory with respect to sequence length n. Doubling the context quadruples the compute. At 1M tokens, naive attention is practically intractable.

Beyond raw complexity, transformers have two other structural properties worth noting:

- **Non-recurrent inference**: At inference time, transformers must recompute or cache all previous keys/values (the KV cache). For autoregressive generation, the KV cache grows linearly with sequence length, consuming significant GPU memory.
- **Positional encoding fragility**: Transformers need explicit positional encodings, and their performance degrades when evaluated on sequences longer than seen during training.

State space models attack both problems at the root by modeling sequences as **linear dynamical systems** rather than pairwise attention graphs.

---

### S4: Structured State Space Sequence Model

S4 (Gu et al., 2021) is the foundational SSM that made this class of models practically viable. The core idea is to model a 1D sequence x(t) → y(t) using a continuous-time state space:

```
h'(t) = A h(t) + B x(t)    # state update
y(t)  = C h(t) + D x(t)    # output
```

Where:
- **h(t)** ∈ ℝ^N is the hidden state
- **A** ∈ ℝ^{N×N} is the state transition matrix
- **B** ∈ ℝ^{N×1}, **C** ∈ ℝ^{1×N}, **D** ∈ ℝ are input/output projections

Discretized with step size Δ (using zero-order hold):

```
h_k = Ā h_{k-1} + B̄ x_k
y_k = C h_k + D x_k

where:
Ā = exp(ΔA)
B̄ = (ΔA)^{-1}(exp(ΔA) - I) · ΔB
```

**The key insight in S4**: the matrix A is parameterized as a **HiPPO matrix** (High-order Polynomial Projection Operators), which is designed to memorize history optimally. Specifically, the HiPPO-LegS matrix projects onto Legendre polynomials, giving the model a principled way to compress long-range history into a fixed-size state.

**Why is this efficient?** SSMs can be computed in two modes:
- **Recurrent mode**: O(N) per step at inference — just update the state vector. No KV cache needed.
- **Convolutional mode**: For training on parallel hardware, unroll the recurrence into a convolution kernel K = (CB̄, CĀB̄, CĀ²B̄, ...) and compute as a single FFT-based convolution in O(n log n).

**S4's limitation**: A is a fixed global matrix — it does not depend on the input content. This makes it less expressive than attention for tasks requiring precise retrieval of specific tokens.

---

### Mamba: Selective State Spaces

Mamba (Gu & Dao, 2023) is the paper that brought SSMs into the LLM conversation. Its core contribution: **input-dependent (selective) state spaces**.

In S4, B, C, and Δ are fixed parameters independent of x. Mamba makes them **functions of the input**:

```
B_k = s_B(x_k)    # input-dependent
C_k = s_C(x_k)    # input-dependent
Δ_k = softplus(s_Δ(x_k))   # input-dependent step size
```

This is a significant departure. Now the model can selectively "focus" on inputs (small Δ → barely update state, large Δ → strongly incorporate new input) and selectively "read" the state (via C_k). This gives it something closer to attention's content-based selection, without the quadratic cost.

**The hardware-aware design** is the other major Mamba contribution. Selective SSMs are no longer purely convolution-friendly (since B, C, Δ change per token), so Gu and Dao wrote a custom CUDA kernel that:
1. Keeps the scan in **SRAM** (fast on-chip memory) rather than HBM
2. Uses **parallel associative scans** to compute the recurrence efficiently on GPU
3. Avoids materializing the full state sequence in HBM

The result: Mamba achieves **5× better throughput** than Transformers at sequence length 2K, and the gap widens with longer sequences.

**Mamba architecture (SSM block)**:
```
Input x → Linear projection (expand 2×) → Split into z, x'
x' → 1D depthwise conv → SiLU activation → SSM (selective scan)
Gated by z via element-wise multiply
→ Linear projection back to d_model
```

This is analogous to a gated MLP/attention block but replaces the attention operation with the selective scan.

---

### When Does Mamba Win vs Transformers?

This is a genuine empirical question, and the answer is nuanced:

| Task Type | Transformer | Mamba |
|-----------|-------------|-------|
| Language modeling (perplexity) | Wins at same param count | Competitive but slightly behind |
| Long-range dependencies (copying, recall) | Strong | Weaker at precise recall |
| Very long sequences (>16K) | Bottlenecked by KV cache | Scales well, recurrent inference |
| Throughput (inference) | Slower due to KV cache | Much faster, especially long ctx |
| DNA/audio/time series | Decent | Stronger (continuous-signal tasks) |

**Mamba's Achilles heel**: in the "associative recall" task (retrieve the value for a specific key from earlier in the sequence), Mamba underperforms attention significantly. The fixed-size state cannot store all past key-value pairs the way attention's full KV cache can. This maps to real-world tasks like in-context learning and few-shot retrieval.

**Key takeaway for interviews**: Mamba is not "better than Transformers." It's a different inductive bias — recurrent, with bounded memory. It excels at modeling continuous signals and generating efficiently at long contexts, but struggles with precise discrete retrieval. The right tool depends on the task.

---

### Hybrid Architectures: Jamba and Zamba

The practical response from the field has been **hybridization** — interleave Mamba layers with attention layers to get the best of both worlds.

**Jamba** (AI21 Labs, 2024):
- Alternates Mamba layers, attention layers, and MoE layers
- Architecture ratio: roughly 1 attention layer per 7 Mamba layers
- 52B total parameters, 12B active (via MoE)
- Context window: 256K tokens, fitting on a single 80GB A100
- Insight: a few attention layers can "fix" precise retrieval; Mamba layers handle long-range compression cheaply

**Zamba** (Zyphra, 2024):
- Uses a single shared attention block applied at multiple points in the network
- Lighter-weight hybridization — reuse attention parameters rather than dedicating full attention layers
- Shows that even very sparse attention coverage is enough to recover recall performance

**Why this matters architecturally**: These hybrids suggest that attention's quadratic cost is mostly wasteful for the bulk of sequence modeling — Mamba-style recurrence handles it cheaper — but you still need attention's precise, content-based selection for certain operations. The optimal architecture likely isn't pure attention or pure SSM, but a learned or designed mix.

---

## 2. Test-Time Compute Scaling

### The Paradigm Shift: From Training Compute to Inference Compute

The dominant scaling narrative through GPT-4 was simple: bigger model + more training data + more training compute = better model. This is the **Chinchilla scaling law** regime.

But in 2024, a new axis emerged: **test-time compute scaling**. The intuition is that some problems benefit from "thinking longer" at inference, not just having more parameters. OpenAI o1 and DeepSeek-R1 are the flagship demonstrations of this idea.

The core claim: given a fixed inference compute budget, a smaller model that thinks more (via extended chain-of-thought and tree search) can outperform a larger model that gives immediate answers.

---

### Chain-of-Thought as Compute Allocation

Chain-of-thought (CoT) prompting (Wei et al., 2022) showed that eliciting reasoning steps dramatically improves performance on multi-step tasks. The mechanism is essentially **extending the computation graph**:

- Without CoT: model computes answer in a single forward pass — O(depth of transformer × sequence length)
- With CoT: model generates T intermediate tokens before the answer, each requiring a forward pass — effectively O(T × depth × length)

This is not just prompting magic. Computationally, each token generation is a new forward pass through the network. CoT transforms a fixed-depth computation into an adaptive-depth one, where harder problems get more "compute steps."

**Training for CoT (RLVR)**: o1 and R1 are trained with reinforcement learning on verifiable rewards (RLVR). The model is rewarded for correct final answers, and long reasoning traces emerge naturally because they correlate with correctness. The key insight: you don't need to supervise the reasoning steps explicitly — just reward the outcome, and the model learns to use intermediate computation effectively.

**DeepSeek-R1** (2025): Open-weight model that matches o1-level reasoning on math/code. Training recipe:
1. Cold start: supervised fine-tuning on small set of human-written long-CoT examples
2. GRPO (Group Relative Policy Optimization): a variant of PPO without a separate value model — compare multiple rollouts per prompt and use relative rewards
3. Rejection sampling + SFT: distill reasoning into a more stable supervised format
4. Final RLHF for helpfulness

The "aha moment" behaviors — where the model spontaneously learns to reconsider, backtrack, and verify — emerge from RL training, not explicit supervision.

---

### Monte Carlo Tree Search (MCTS) for Reasoning

MCTS is a search algorithm from game-playing AI (AlphaGo) that builds a tree of possible moves, using rollouts to estimate the value of unexplored nodes. Applied to language model reasoning:

```
Tree structure:
- Node = partial reasoning trace (prefix)
- Edge = next reasoning step (token or sentence)
- Value = estimated probability of reaching correct answer from this node
```

**MCTS for LLM reasoning loop:**
1. **Selection**: traverse tree using UCB (Upper Confidence Bound) to balance exploration vs exploitation
2. **Expansion**: generate K candidate next steps from current node using the LLM
3. **Simulation/Rollout**: complete the reasoning trace and check final answer
4. **Backpropagation**: update node values based on rollout outcome

```
UCB score = Q(s,a) + c · sqrt(ln N(s) / N(s,a))

where:
Q(s,a) = average reward from taking action a at state s
N(s)   = visit count for state s
N(s,a) = visit count for edge (s,a)
c      = exploration constant
```

**Practical variants**: Full MCTS is expensive (many rollouts). More practical variants:
- **Beam search with a verifier**: maintain top-k reasoning paths, prune by a trained reward model
- **DVTS (Diverse Verifier Tree Search)**: diversify the beam to avoid mode collapse
- **Step-level beam search**: evaluate and prune at each reasoning step rather than only at the end

---

### Verification vs Generation: The Asymmetry

A key structural insight: **verification is often easier than generation**. It's harder to write a correct proof from scratch than to check whether a given proof is valid. This asymmetry is exploited by test-time compute methods.

**Generator-verifier separation**:
- **Generator**: LLM that produces candidate answers/reasoning traces
- **Verifier (reward model)**: trained model that scores whether a trace is correct

If the verifier is reliable, you can generate N candidate answers and pick the best-scored one — trading inference compute for accuracy without retraining the generator.

**The verification bottleneck**: verifiers are trained on human labels and can be gamed. Process reward models (PRMs) try to address this by evaluating at every step.

---

### Best-of-N and Process Reward Models

**Best-of-N (BoN)**: generate N independent completions, pick the one that scores highest under a reward model. Empirically, BoN scales very well — doubling N gives consistent accuracy improvements on math benchmarks, up to surprisingly large N (~1000).

**Outcome Reward Models (ORMs)**: score only the final answer. Simple but prone to rewarding shortcuts — a model can reach the right answer via wrong reasoning.

**Process Reward Models (PRMs)**: score each intermediate reasoning step. More expensive to train (requires step-level human labels or automated verification) but much harder to game. OpenAI's PRM800K dataset (800K step-level labels) is the canonical resource here.

**PRM training objective** (simplified):
```
For reasoning trace [s_1, s_2, ..., s_T, answer]:
PRM predicts P(step s_t is correct | s_1, ..., s_t)
Trained on binary labels per step via cross-entropy
```

**BoN with PRM**: instead of scoring whole traces, aggregate PRM scores across steps. Several aggregation strategies work — min score (most conservative), product, or a learned aggregator.

---

### Smaller Model + More Compute vs Larger Model

The fundamental trade-off that test-time compute scaling surfaces:

| Approach | Training Cost | Inference Cost | Latency | Accuracy |
|----------|--------------|----------------|---------|----------|
| Larger pretrained model | Very high | Moderate (per token) | Low | High on easy tasks |
| Smaller model + BoN-1000 | Moderate | Very high | High | Competitive on hard tasks |
| Smaller model + PRM + beam search | Moderate + PRM training | High | Moderate | Strong on structured reasoning |

**The inflection point**: for hard mathematical reasoning, a smaller model with extended thinking can match a 10× larger model. But for factual recall and broad knowledge tasks, model size still dominates.

**Practical implication**: inference compute is often cheaper per unit than training compute (can be batched across users, doesn't require gradient storage). This changes the economic calculus — it may be cheaper to run a small model with aggressive test-time compute than to train and serve a massive model.

---

## 3. Mixture of Experts (MoE)

### The Core Idea

Dense neural networks apply every parameter to every input token. MoE breaks this: the network has N "expert" subnetworks (usually FFN layers), and a learned **router** activates only k of them per token. Total parameters scale with N, but FLOPs per forward pass scale only with k.

```
MoE layer output for token x:
y = Σ_{i ∈ Top-k(G(x))} G_i(x) · E_i(x)

where:
G(x) = softmax(W_g · x)     # router logits
E_i(x) = FFN_i(x)           # i-th expert FFN
Top-k(G(x))                 # indices of k largest gate values
```

The FFN sublayers in a standard transformer are the most parameter-heavy components (typically 2/3 of parameters). Replacing them with MoE layers gives a multiplier on effective parameter count without proportional compute cost.

---

### Router Mechanisms

**Top-k routing** (most common, k=1 or k=2):
- Compute gate logits for all N experts
- Activate the top-k experts
- Weight their outputs by (optionally normalized) gate values

**Load balancing loss**: naive top-k routing leads to **routing collapse** — the router always sends tokens to the same few experts, leaving others unused. To prevent this, add an auxiliary load balancing loss:

```
L_balance = α · N · Σ_i f_i · P_i

where:
f_i = fraction of tokens routed to expert i
P_i = mean gate probability for expert i
N   = number of experts
α   = balancing coefficient (typically 0.01–0.1)
```

This encourages uniform utilization while still allowing learned routing.

**Expert capacity**: in batch processing, set a capacity limit C per expert (tokens per batch). If an expert is over capacity, excess tokens are "dropped" (or handled by a residual path). This prevents compute imbalance but introduces information loss.

**Alternative routing mechanisms:**
- **Expert Choice routing** (Zhou et al., 2022): flip the perspective — each expert chooses its top-k tokens rather than each token choosing experts. Guarantees perfect load balance but breaks the autoregressive property (expert choices depend on future tokens in the batch).
- **Soft routing / Mixture of Softmaxes**: route to all experts with soft weights, no hard top-k. Avoids load balancing issues but loses compute sparsity.
- **Hash routing**: deterministic assignment based on token hash. Perfectly balanced, zero routing overhead, but ignores content — each token always goes to the same experts regardless of meaning.

---

### Sparse vs Dense: The Tradeoff Space

| Property | Dense FFN | Sparse MoE (top-2, N=8) |
|----------|-----------|-------------------------|
| Parameters | P | 8P (FFN portion) |
| FLOPs per token | F | 2F/8 = F/4 (roughly) |
| Memory | Low | High (all experts must fit) |
| Communication overhead | None | All-to-all across devices |
| Parallelism | Tensor/pipeline | Expert parallelism needed |

**Key insight**: MoE is a parameter efficiency play, not a compute efficiency play. You get a "free" parameter increase, which translates to higher model capacity (and usually quality) at the same compute budget. But you pay in memory and communication.

**Expert parallelism**: experts are distributed across devices. Each token is sent to its expert's device, output is returned. This requires **all-to-all collective communication** — every device sends data to every other device. At scale, this communication can become the bottleneck (especially on networks with limited bisection bandwidth like inter-node NVLink vs InfiniBand).

---

### Switch Transformer

Switch Transformer (Fedus et al., 2021) simplified MoE by using **top-1 routing** (k=1). Despite conventional wisdom that k=2 is needed for stability, top-1 routing works if combined with:
1. Careful initialization (router weights initialized to small values)
2. Selective precision: keep router computations in float32 even in mixed-precision training
3. The load balancing loss described above

Switch Transformer scaled to 1.6T parameters (sparse) and matched a dense T5-11B model at 7× fewer FLOPs per step, or significantly outperformed it at equal FLOPs.

---

### Mixtral 8x7B

Mixtral (Mistral AI, 2023) is the most widely-used open MoE model. Architecture:
- 8 experts per MoE layer, top-2 routing
- 46.7B total parameters, 12.9B active per token
- Only MoE in the FFN sublayers; attention layers are standard dense
- Sliding window attention for efficient long-context handling

**Performance**: Mixtral 8x7B outperforms Llama 2 70B on most benchmarks at roughly equal inference cost (since only 12.9B parameters are active per token). This is MoE's value proposition in practice.

**GPT-4 (rumored MoE)**: multiple sources indicate GPT-4 is a MoE model with ~8 experts and ~220B active parameters (out of ~1.8T total). This explains GPT-4's quality vs cost profile. While not officially confirmed, the architecture matches what the MoE scaling literature would predict for its capability tier.

---

### Training Instability Challenges

MoE training is notoriously fragile compared to dense training. Common failure modes:

**Router instability**: gate logits can saturate early, causing the router to lock onto suboptimal assignments before experts have learned meaningful specializations. Mitigation: jitter noise during training (`G(x) = softmax(W_g · x + ε)` where ε ~ N(0, noise_std²)).

**Gradient spikes**: the discrete top-k operation creates discontinuous gradients. The straight-through estimator and careful learning rate scheduling help.

**Expert collapse**: all tokens route to one expert, others receive no gradient and stagnate. The load balancing loss is the primary mitigation, but tuning the coefficient α is critical — too small and collapse occurs, too large and it overwhelms the task loss.

**Communication deadlocks**: at large scale, all-to-all communication can deadlock or create severe stragglers. This requires careful capacity buffer sizing and fallback paths.

---

### Expert Collapse and Prevention

Expert collapse occurs when routing entropy is too low — the distribution over experts is too peaked. Diagnostics:

```python
# Monitor routing entropy during training
routing_entropy = -sum(p_i * log(p_i) for p_i in expert_utilization)
# Target: close to log(N) (maximum entropy = uniform routing)
```

**Prevention strategies:**
1. **Load balancing loss** (required baseline)
2. **Router z-loss** (from ST-MoE, Zoph et al. 2022): penalize large router logit magnitudes
   ```
   L_z = β · (1/B) Σ_x (log Σ_i exp(h_i(x)))²
   ```
3. **Expert dropout**: randomly drop experts during training, forcing others to cover
4. **Auxiliary expert loss**: add small reconstruction loss on each expert independently
5. **Curriculum MoE**: start training with fewer experts (or dense), then gradually introduce more

---

## 4. Long Context & Efficient Attention

### Context Window Evolution

The context window in transformer LLMs has expanded dramatically:

| Year | Model | Context Window |
|------|-------|---------------|
| 2019 | GPT-2 | 1,024 tokens |
| 2020 | GPT-3 | 2,048 tokens |
| 2022 | PaLM | 2,048 tokens |
| 2023 | Claude 2 | 100,000 tokens |
| 2023 | GPT-4 Turbo | 128,000 tokens |
| 2024 | Gemini 1.5 Pro | 1,000,000 tokens |
| 2024 | Claude 3 (Haiku/Sonnet/Opus) | 200,000 tokens |

This expansion required innovations at multiple levels: positional encoding methods that generalize, attention algorithms that fit in memory, and training strategies to handle ultra-long sequences.

---

### Flash Attention 2 and 3

Standard attention materializes the full N×N attention matrix in HBM (high-bandwidth memory), requiring O(N²) memory. Flash Attention (Dao et al., 2022) rewrites attention to be **IO-aware**:

**Flash Attention algorithm sketch:**
1. Tile Q, K, V into blocks that fit in SRAM
2. Compute attention block-by-block, maintaining running softmax statistics (max and sum) for numerical stability
3. Never materialize the full N×N matrix in HBM
4. Fuse backward pass to avoid re-reading Q, K, V from HBM

```
Memory complexity: O(N)  vs O(N²) for standard attention
IO complexity: O(N² / M) where M = SRAM size
(reads K, V O(N/block_size) times each)
```

**Flash Attention 2** (Dao, 2023) adds:
- Parallelization across the sequence dimension (not just batch/head dimensions)
- Better work partitioning between GPU warps to reduce synchronization overhead
- ~2× speedup over FA1, achieving ~70% of theoretical GPU FLOP utilization

**Flash Attention 3** (Shah et al., 2024), targeting H100/Hopper:
- Exploits WGMMA (Warpgroup Matrix Multiply Accumulate) instructions
- Overlaps softmax and GEMM via pipelining — while computing attention scores, simultaneously start loading next K/V tile
- Asynchronous data movement via TMA (Tensor Memory Accelerator)
- ~1.5–2× speedup over FA2 on H100

---

### Ring Attention

Ring Attention (Liu et al., 2023) solves a different problem: how to compute attention when the full sequence doesn't fit on a single GPU at all.

**Setup**: distribute the sequence across D devices in a ring topology. Each device holds a chunk of Q, K, V of size N/D.

**Algorithm**:
1. Each device computes local attention for its Q chunk against its local K, V chunk
2. Devices pass their K, V chunks to the next device in the ring (while computing)
3. After D-1 passes, each device has seen all K, V blocks and can compute full attention for its Q chunk

**Key insight**: the communication (sending K/V blocks around the ring) overlaps with computation (computing attention on the K/V block you currently hold), hiding the communication latency.

**Memory**: each device only needs memory for N/D tokens of Q, K, V, plus one K/V chunk in flight. This makes attention on million-token sequences feasible across many devices.

---

### RoPE Scaling: YaRN and LongRoPE

Most modern LLMs use **RoPE** (Rotary Position Embedding, Su et al., 2021). RoPE encodes position by rotating Q and K vectors:

```
For position m and dimension pair (2i, 2i+1):
[q_{2i}, q_{2i+1}] → [q_{2i}cos(mθ_i) - q_{2i+1}sin(mθ_i),
                       q_{2i}sin(mθ_i) + q_{2i+1}cos(mθ_i)]

where θ_i = 10000^{-2i/d}
```

The problem: a model trained with max length L has only seen positions 0...L-1. At position m > L, the rotation angles are "out of distribution" and performance degrades sharply.

**Simple positional interpolation**: scale positions down — map position m to m × (L / L_new). This keeps all angles in-distribution but reduces positional resolution. Fine-tuning is still needed but minimal.

**YaRN** (Peng et al., 2023): more sophisticated — different frequency dimensions of RoPE have different optimal scaling strategies:
- High-frequency components (small θ): don't scale (they're already dense, scaling would hurt)
- Low-frequency components (large θ): interpolate
- Mid-frequency: a ramp function between no-scaling and interpolation
- Also scales the attention temperature by √(log(L_new) / log(L)) to prevent attention entropy collapse

```python
# YaRN position scaling (simplified)
def yarn_scale(dim_idx, d_model, base=10000, scale=8):
    theta = base ** (-2 * dim_idx / d_model)
    low_freq_factor = 1  # below this wavelength, interpolate
    high_freq_factor = 4  # above this, don't scale
    
    if theta > high_freq_factor:  # high freq: no scaling
        return 1.0
    elif theta < low_freq_factor:  # low freq: full interpolation
        return scale
    else:  # smooth ramp
        return 1 + (scale - 1) * (theta - high_freq_factor) / (low_freq_factor - high_freq_factor)
```

**LongRoPE** (Ding et al., 2024): searches for per-dimension scaling factors via evolutionary optimization rather than a fixed formula. Claims 2M token context with minimal perplexity degradation.

---

### Lost-in-the-Middle Problem

Liu et al. (2023) showed empirically that LLMs struggle to use information from the **middle** of long contexts. Performance is highest when relevant information is at the beginning or end of the context (primacy and recency effects), and degrades sharply for information in the middle.

**Intuition**: attention score distributions in long-context settings tend to concentrate on initial tokens (due to attention sink patterns) and recent tokens (locality bias). Middle tokens receive lower cumulative attention.

**Attention sinks**: Xiao et al. (2023) showed that certain "sink tokens" (like [BOS] or punctuation) attract disproportionate attention. Removing them causes performance collapse, even though they carry no semantic content — they serve as "attention sponges" that absorb attention probability mass that would otherwise be noise.

**Practical implications**:
- For RAG: don't just paste retrieved context arbitrarily — put the most relevant chunk at the beginning or end
- For long-document QA: consider re-ranking retrieved passages specifically for position-robustness
- For evaluation: test retrieval at different positions when benchmarking long-context models

---

### Retrieval vs Context: When to Use Which

The million-token context window raises a genuine question: do you still need RAG?

**Use retrieval (RAG) when:**
- Your knowledge base is much larger than any context window (enterprise docs, full codebases)
- Knowledge changes frequently (news, live data) — context is static, retrieval is dynamic
- You need citation/attribution — retrieval gives you provenance
- Cost matters — serving N million tokens per query is expensive
- Precision matters — dense retrieval can surface the right chunk even from billions of docs

**Use long context when:**
- Relationships span the entire document (legal contracts, books, full codebases)
- You need the model to reason over all parts simultaneously (no known retrieval boundary)
- Latency matters more than cost (one big context vs retrieval + rerank pipeline)
- The relevant content is not pre-indexable (ad-hoc documents, pastes)

**Hybrid**: many production systems use retrieval to identify the most relevant chunks, then provide them in a moderately long context (16K–128K) for the model to reason over. This combines retrieval's scalability with context's holistic reasoning.

---

## 5. Synthetic Data & Data Flywheels

### The Data Scarcity Problem

Internet-scale pretraining has hit a ceiling: we've essentially trained on all publicly available high-quality text. Estimates suggest we'll exhaust "unique quality tokens" within a few years at current scaling rates (Villalobos et al., 2022). Synthetic data is the primary solution being pursued.

---

### Self-Play and Rejection Sampling

**Rejection sampling fine-tuning (RFT)**: generate N solutions per problem using the current model, keep only the correct ones, fine-tune on the filtered set. This is a simple but powerful bootstrapping loop:

```
1. Start with base model M_0
2. For each problem p in dataset D:
   a. Sample N solutions: {s_1, ..., s_N} ~ M_0(p)
   b. Verify each solution (math: check answer, code: run tests)
   c. Keep correct solutions: D_correct = {s_i : verify(s_i, p) = True}
3. Fine-tune M_0 on D_correct to get M_1
4. Repeat with M_1 as the new generator
```

This is a **data flywheel**: a better model generates better data, which trains an even better model. The key requirement is a reliable verifier (ground-truth answers for math, test suites for code).

**Self-play** (broader): pitting models against themselves in adversarial or cooperative settings to generate training data. Examples:
- **Debate**: one model argues, another rebuts — human evaluates which argument is more compelling
- **Red-teaming**: adversary model generates attacks, defender model learns to resist them
- **Constitutional AI**: model critiques its own outputs and revises them

---

### Constitutional AI (CAI) and RLAIF

Constitutional AI (Bai et al., Anthropic 2022) is a method for training helpful and harmless models without relying entirely on human feedback for harmlessness:

**Phase 1 — Supervised Learning from AI Feedback (SL-CAI):**
1. Sample harmful responses from an initial helpful model
2. Ask the model to revise its response according to a **constitution** (a set of principles: "don't provide instructions for harmful activities," "be honest," etc.)
3. Fine-tune on the revised (harmless) responses

**Phase 2 — RL from AI Feedback (RLAIF):**
1. Generate pairs of responses to prompts
2. Ask a feedback model (also using the constitution) to evaluate which response is better
3. Train a preference model on these AI-generated labels
4. Use PPO against this preference model — same as RLHF but feedback is from an AI, not humans

**Why it matters**: human labeling for harmlessness is expensive and inconsistent. RLAIF scales better and can be applied iteratively. The risk is that the feedback model inherits the same biases as the generator — you're in a loop, not getting ground truth.

---

### Phi Models: Quality Over Quantity

Microsoft's Phi series (Gunasekar et al., 2023) demonstrated that **data quality matters more than quantity at smaller scales**:

- **Phi-1** (1.3B parameters): trained primarily on "textbook quality" code — synthetically generated, pedagogically structured coding examples. Outperformed models 10× larger on HumanEval despite having a tiny dataset.
- **Phi-1.5 / Phi-2** (2.7B): extended to general reasoning with "textbook quality" web data filtered and augmented by GPT-4
- **Phi-3** (3.8B, 7B, 14B): instruction-tuned on synthetic data, competitive with Llama 3 70B on many benchmarks

**Synthetic data recipe for Phi:**
1. Generate diverse educational content (step-by-step explanations, worked examples) using a large teacher model
2. Mix with filtered web data (Common Crawl filtered by an educational quality classifier)
3. Heavy deduplication to avoid memorization
4. Curriculum ordering: simpler concepts before complex ones

The underlying insight: models learn better from didactically structured data than from raw internet text, even if raw internet text is orders of magnitude larger.

---

### Curriculum Learning with Synthetic Data

**Curriculum learning** (Bengio et al., 2009): present training examples in a meaningful order, moving from easy to hard. With synthetic data, you can control difficulty precisely:

**Difficulty axes for synthetic data:**
- Reasoning chain length (number of steps)
- Number of constraints / variables
- Domain coverage (single vs multi-step cross-domain)
- Noise level (distractor information)

**Practical curriculum for math reasoning:**
1. Simple arithmetic → multi-step arithmetic → algebra → calculus notation
2. At each stage: use models that can solve the problems to generate worked solutions
3. Verify all solutions before adding to training set
4. Gradually reduce verification strictness as the student model improves

**Self-curriculum**: as the model improves, it can generate progressively harder problems for itself. The trick is measuring difficulty — use the current model's success rate as a proxy (problems where the model succeeds ~50–80% of the time are in the "zone of proximal development").

---

### Risks: Model Collapse and Echo Chambers

**Model collapse** (Shumailov et al., 2023): when models train on AI-generated data that was itself generated by earlier models, the distribution of the training data degrades progressively:
- Early models trained on human data learn the full distribution
- Models trained on AI-generated data see a distribution filtered through the previous model's biases
- Each generation, low-probability (diverse, creative) content is further suppressed
- Eventually, the model "collapses" to a narrow, repetitive distribution

**Formal intuition:**
```
Human data: p(x)
Model M_0 trained on p(x): approximates p(x)
M_0-generated data: q_0(x) ≈ p(x) but with tails trimmed
M_1 trained on q_0(x): approximates q_0(x) with further trimmed tails
...
M_n: increasingly degenerate, low-variance distribution
```

**Mitigations:**
- Always mix a significant fraction of real human data (do not train purely on synthetic)
- Use diverse sampling strategies (high temperature, diverse prompts) during generation
- Monitor distribution statistics (vocabulary diversity, n-gram entropy) across training generations
- Use verification to filter for quality, not just correctness — avoid over-filtering

**Echo chambers**: if your AI feedback model and your generator share similar biases (same base model, same RLHF pipeline), AI feedback can reinforce those biases rather than correcting them. This is a fundamental limit of purely closed-loop systems.

---

## 6. Multimodal Foundation Models

### Vision-Language Models (VLMs)

**CLIP** (Contrastive Language-Image Pre-training, Radford et al., 2021):
- Train image encoder and text encoder jointly via contrastive loss on 400M image-text pairs
- Objective: maximize cosine similarity of (image, text) pairs; minimize for non-pairs
  ```
  L = -log(exp(sim(I_i, T_i)/τ) / Σ_j exp(sim(I_i, T_j)/τ))
  ```
- Creates a shared embedding space where semantically related images and text are close
- Zero-shot classification: compute similarity of image embedding to text embeddings of class descriptions

**BLIP / BLIP-2** (Li et al., 2022/2023):
- BLIP-2 introduces the **Q-Former**: a lightweight transformer that bridges a frozen image encoder and a frozen LLM
- Q-Former has N learnable "query" tokens that attend to image features and extract the most relevant information for language
- Only the Q-Former is trained — keeps both the vision encoder and LLM frozen, dramatically reducing training cost
- Bootstrapping: pre-train on image-text matching + captioning + VQA objectives, then connect to LLM

**LLaVA** (Liu et al., 2023):
- Even simpler: project image features (from CLIP ViT) directly into the LLM's embedding space via a linear projection or small MLP
- Generate instruction-following data synthetically: use GPT-4 to write conversations about images based on COCO captions
- Fine-tune LLaMA/Vicuna on this data
- Surprisingly competitive at much lower training cost than BLIP-2

**GPT-4V / Claude 3 Vision**: production VLMs with proprietary architectures. Key capabilities beyond BLIP-2/LLaVA: fine-grained OCR, spatial reasoning, diagram understanding, multi-image reasoning.

---

### Image Tokenization: VQVAE and dVAE

To use images as inputs to transformers in a "token" sense (as opposed to continuous features), images need to be discretized.

**VQVAE** (van den Oord et al., 2017):
- Encoder maps image to latent grid z_e(x)
- Vector quantization: each spatial position is mapped to nearest code in a learned codebook C of size K
  ```
  z_q(x) = arg min_{c_k ∈ C} ||z_e(x) - c_k||_2
  ```
- Decoder reconstructs image from quantized latents
- Training trick: straight-through estimator for gradients through the discrete argmin
- Result: image → sequence of discrete tokens from vocabulary of size K

**dVAE** (discrete VAE, used by DALL-E 1):
- Variational formulation: instead of hard argmin, use Gumbel-softmax to sample from a soft categorical distribution over codebook entries
- Temperature annealing: start with soft (differentiable) sampling, gradually approach hard discrete sampling
- Scales to larger codebooks and images

**Why this matters**: once images are tokenized, you can train a GPT-style autoregressive model over image tokens directly (as DALL-E 1 did). Modern image generation models (DALL-E 3, Stable Diffusion XL) use diffusion rather than autoregressive token generation, but the tokenization step persists as the first stage of many VLM architectures.

---

### Audio and Video Modalities

**Audio**:
- Raw waveforms → spectrograms (STFT) → 2D feature maps treated like images
- **Whisper** (Radford et al., 2022): sequence-to-sequence model on log-mel spectrograms, trained on 680K hours of internet audio with weak supervision (automatically transcribed text)
- **EnCodec / SoundStream**: neural audio codecs that compress audio to discrete tokens — enable LLMs to model audio autoregressively (AudioPaLM, MusicGen)

**Video**:
- Spatial + temporal dimensions make video vastly more expensive than images
- Key challenge: 30fps × 3600s × 1080p = enormous data
- **Temporal compression**: sample sparse frames (1fps or less), or use optical flow to represent motion compactly
- **Video Transformers** (ViViT, TimeSformer): factorize spatial and temporal attention — attend across space within a frame, then across time across frames
- **Sora** (OpenAI, 2024): trains a diffusion transformer on continuous spacetime patches ("spacetime tokens"), can generate coherent minute-long videos with physics plausibility

---

### Unified vs Modality-Specific Encoders

**Two architectural philosophies:**

**Unified encoders** (e.g., Perceiver, Flamingo, Gemini): a single backbone processes all modalities
- Pros: shared representations, potential cross-modal transfer, simpler serving
- Cons: harder to train (modalities interfere), may underfit modality-specific structure

**Modality-specific + fusion** (e.g., LLaVA, BLIP-2, Whisper + LLM): separate specialist encoders, fused at a late stage
- Pros: leverage existing specialist checkpoints, easier to train, clear modularity
- Cons: no deep cross-modal interaction, fusion layer may be a bottleneck

**Trend**: most practical production systems (GPT-4o, Gemini) use some form of late fusion with strong modality-specific encoders. True "native multimodality" (processing all modalities in a unified token stream from the beginning) is a research direction but not yet the dominant deployment paradigm.

---

## 7. Efficient Fine-tuning Evolution

### Why Parameter-Efficient Fine-tuning Matters

Full fine-tuning of a 70B model requires storing:
- Model weights: 70B × 2 bytes (bf16) = 140GB
- Gradients: 140GB
- Adam optimizer states (momentum + variance): 560GB
- Total: ~840GB minimum — 10+ A100s just for optimizer state

PEFT methods reduce this to fine-tune a small number of parameters while keeping most of the model frozen.

---

### LoRA and Its Evolution

**LoRA** (Low-Rank Adaptation, Hu et al., 2021):
For a weight matrix W ∈ ℝ^{d×k}, instead of updating W directly, train two low-rank matrices:
```
W' = W + ΔW = W + B · A

where:
A ∈ ℝ^{r×k}, B ∈ ℝ^{d×r}, r << min(d, k)
A initialized from N(0, σ²), B initialized to 0
(so ΔW = 0 at initialization, full model behavior preserved)
```

Scaling factor α/r applied to ΔW for stability. Trainable parameters: r(d+k) vs d×k for full fine-tuning. At r=16, d=k=4096: 131K vs 16.7M parameters.

At inference: merge W + BA back into W — zero additional latency.

**LoRA+** (Hayou et al., 2024): different learning rates for A and B matrices. B (output matrix) should have a higher learning rate than A (input matrix) because of the asymmetric gradient flow through the product BA. Provides meaningful improvement with no architecture changes.

**DoRA** (Weight-Decomposed Low-Rank Adaptation, Liu et al., 2024):
Decompose the weight update into **magnitude** and **direction** components separately:
```
W' = (||W + BA||_c / ||W||_c) · (W + BA) / ||W + BA||_c
   = m · (W + BA) / ||W + BA||_c

where m = ||W||_c is a learnable scalar per column, ||·||_c is column-wise norm
```
By training magnitude and direction independently, DoRA more closely mimics the update pattern of full fine-tuning, achieving better performance especially in low-rank regimes.

**QLoRA** (Dettmers et al., 2023):
Combines LoRA with 4-bit quantization of the frozen base model:
1. Quantize base model to NF4 (NormalFloat4) — a data type optimized for normally distributed weights
2. Use double quantization (quantize the quantization constants too)
3. Apply LoRA adapters in bf16 on top of the 4-bit base
4. Use paged optimizers to manage GPU memory spikes

Result: fine-tune a 65B model on a single 48GB GPU. The quality degradation vs full bf16 fine-tuning is minimal (within 1% on benchmarks).

---

### Adapter Layers, Prefix Tuning, Prompt Tuning

**Adapter layers** (Houlsby et al., 2019):
Insert small bottleneck networks inside each transformer layer:
```
Adapter(x) = x + W_up · ReLU(W_down · LN(x))
where W_down ∈ ℝ^{d×r}, W_up ∈ ℝ^{r×d}, r << d
```
Only adapter parameters are trained. Adapter adds ~2r² parameters per layer, plus minor latency at inference (cannot be merged like LoRA).

**Prefix tuning** (Li & Liang, 2021):
Prepend a sequence of K learnable "virtual tokens" to the keys and values at each attention layer:
```
Attention(Q, [P_K; K], [P_V; V])
where P_K, P_V ∈ ℝ^{K×d} are learned prefix matrices
```
The model never sees these as real tokens but attends to them as additional context that steers behavior. No inference latency overhead (prefix is just extended K, V).

**Prompt tuning** (Lester et al., 2021):
Even simpler — prepend soft prompt tokens only to the input embedding layer (not to every attention layer). Only effective at large model scales (>10B parameters); underperforms at small scales.

**Comparison at r=8:**

| Method | Trainable params | Inference overhead | Performance |
|--------|-----------------|-------------------|-------------|
| Full FT | 100% | None | Best |
| LoRA (r=8) | ~0.1% | None (merge) | Near-full |
| Adapters (r=8) | ~0.3% | ~5-10ms/layer | Near-full |
| Prefix (K=10) | ~0.05% | Slight KV overhead | Good |
| Prompt tuning | ~0.01% | Minimal | Weaker at small scale |

---

### Full Fine-tuning with DeepSpeed ZeRO

For cases requiring full fine-tuning (catastrophic forgetting prevention, domain adaptation), DeepSpeed's ZeRO (Zero Redundancy Optimizer) makes it feasible:

**ZeRO Stage 1**: Partition optimizer states across data-parallel GPUs. Each GPU holds 1/N of optimizer state (momentum, variance). Reduces optimizer memory by N×.

**ZeRO Stage 2**: Additionally partition gradients. Each GPU accumulates and holds 1/N of the gradient. Reduces gradient memory by N× on top of Stage 1.

**ZeRO Stage 3**: Additionally partition model parameters. Each GPU holds 1/N of the model weights. Requires all-gather before each forward/backward pass, but enables fitting enormous models across many GPUs.

**ZeRO-Offload**: offload optimizer states and gradients to CPU RAM, using CPU compute for Adam updates. Significantly slower but allows training large models on fewer GPUs.

**ZeRO-Infinity**: offload to NVMe SSD. Can train trillion-parameter models on a single node (very slow, but feasible for research).

---

## 8. AI Agents & MCP

### Tool Use Evolution: From Function Calling to MCP

**Function calling** (OpenAI, 2023): provide the LLM with a JSON schema of available functions; model outputs a structured function call; application executes it and returns results. This is stateless and application-specific — each application defines its own tools.

**Model Context Protocol (MCP)** (Anthropic, 2024): a standardized open protocol for tool/resource provision to LLMs:
- Defines a standard client-server architecture: **MCP hosts** (LLM applications) connect to **MCP servers** (tool providers)
- Servers expose: **Tools** (callable functions), **Resources** (readable data), **Prompts** (reusable prompt templates)
- Transport: JSON-RPC over stdio or SSE (Server-Sent Events)
- Key benefit: a single MCP server (e.g., a PostgreSQL MCP server) works with any MCP-compatible host (Claude Desktop, Cursor, custom apps)

This is analogous to LSP (Language Server Protocol) for code editors — separates tool logic from LLM client logic, enabling a rich ecosystem of reusable tools.

---

### Computer Use Agents

Claude's Computer Use API (October 2024) enables models to interact with computers as a human would:
- Screenshot → model describes state and plans action
- Available actions: mouse click, keyboard type, scroll, screenshot
- The model issues action commands; the orchestrator executes them and returns updated screenshots

**Challenges specific to computer use:**
- **Latency**: each action requires a round-trip to the model; complex tasks require hundreds of actions
- **Error recovery**: models must detect when an action failed and retry or change strategy
- **Long-horizon planning**: breaking a high-level task ("book a flight") into atomic GUI actions requires multi-level planning
- **Safety**: models can take destructive actions (delete files, send emails) — sandboxing and human oversight are critical

---

### Multi-Agent Coordination

Single agents have limited context and cannot parallelize. Multi-agent systems address this:

**Architectures:**
- **Orchestrator-subagent**: a coordinator agent breaks down tasks and delegates to specialist subagents, collects results
- **Peer-to-peer**: agents communicate directly, each with a specific role (critic, executor, planner)
- **Swarm**: many identical agents in parallel, results aggregated (useful for BoN-style sampling)

**Coordination mechanisms:**
- **Shared memory / blackboard**: agents read/write to a shared state (database, document)
- **Message passing**: agents communicate via structured messages (tool calls between agents)
- **Voting / debate**: multiple agents produce outputs, a judge selects or synthesizes

**Key challenges:**
- **Context contamination**: long conversation histories in sub-agents consume context quickly
- **Error propagation**: mistakes in early agents cascade through the pipeline
- **Reliability**: each agent has some error rate ε; N-agent pipeline has compound error ≈ 1-(1-ε)^N
- **Trust**: should an orchestrator trust subagent outputs? Verification loops add latency

---

### Memory Architectures for Agents

Agents need multiple types of memory:

```
┌────────────────────────────────────────────────────────┐
│                    Agent Memory Types                   │
├────────────────┬───────────────────────────────────────┤
│ In-context     │ Current conversation, recent actions  │
│ (working)      │ Fast access, limited capacity (128K)  │
├────────────────┼───────────────────────────────────────┤
│ External       │ Vector DB, key-value store, SQL DB     │
│ (long-term)    │ Unlimited capacity, slower access      │
├────────────────┼───────────────────────────────────────┤
│ Procedural     │ Baked into model weights via fine-     │
│ (parametric)   │ tuning; no explicit retrieval needed  │
├────────────────┼───────────────────────────────────────┤
│ Episodic       │ Past interaction summaries; compressed │
│ (summarized)   │ to fit in context or stored externally │
└────────────────┴───────────────────────────────────────┘
```

**Practical memory management for long-running agents:**
1. **Rolling summarization**: every K interactions, summarize the oldest K/2 interactions into a compact summary; drop raw history
2. **Semantic compression**: embed interactions and cluster — store cluster centroids as compressed memory
3. **Retrieval-augmented memory**: store all interactions in a vector DB; retrieve the most relevant ones at each step
4. **Memory consolidation**: periodically fine-tune the model on interaction history (expensive but persistent)

---

## 9. AI Safety Frontiers

### Constitutional AI and RLAIF

(See also Section 5 for the CAI training procedure)

Constitutional AI represents a shift from **direct human feedback** to **principle-guided AI feedback**. The constitution acts as a compressed, auditable representation of human values that can be applied at scale.

**Key properties of a good constitution:**
- Covers both positive behaviors (be helpful, accurate) and negative constraints (don't assist harmful activities)
- Specific enough to resolve ambiguous cases
- Internally consistent — principles don't contradict each other
- Generalizes — doesn't just cover known edge cases but new ones too

**Limitations**: the constitution reflects the values of its authors. Anthropic's constitution embeds their views about what's helpful and harmless. This is more transparent than RLHF (where annotator biases are implicit) but still not "value-neutral."

---

### Mechanistic Interpretability

Mechanistic interpretability is the effort to reverse-engineer exactly what neural networks compute — not just what they output, but the specific computational circuits that produce those outputs.

**Superposition hypothesis** (Elman, Elhage et al., Anthropic):
Neural networks with d hidden dimensions can represent far more than d features by storing them in superposition — multiple features share dimensions, encoded as nearly-orthogonal vectors:

```
For n features with d dimensions (n >> d):
Each feature i is represented as a vector f_i ∈ ℝ^d
Features are approximately orthogonal: f_i · f_j ≈ 0 for i ≠ j
But only approximately — interference between features creates noise

Recall: if features are sparse (most features are 0 for a given input),
then interference is rare and the scheme works despite n >> d
```

**Why this matters**: it explains why neural network internals are hard to interpret — individual neurons don't correspond to single features; features are distributed across neurons in non-obvious ways.

**Circuits-based interpretability**: Anthropic's circuits work (Cammarata et al., Elhage et al.) traced specific computations:
- Curve detectors in vision models: specific neurons that respond to curves at particular orientations
- Induction heads in language models: two-layer attention circuits that perform in-context learning (copy previous completions of the same pattern)
- Indirect object identification circuit: the set of attention heads in GPT-2 that identify "Mary gave John the book → John received it from Mary"

**Sparse Autoencoders (SAEs)** for monosemanticity: Anthropic's 2023–2024 work trains sparse autoencoders on model activations to decompose superposed representations into (approximately) monosemantic features:
```
SAE: activation a ∈ ℝ^d → features f = ReLU(W_enc(a - b_dec)) ∈ ℝ^m
     reconstruction: â = W_dec f + b_dec
     loss: ||a - â||² + λ||f||₁   (reconstruction + sparsity)
```
The sparsity penalty encourages each feature to activate rarely and independently. Result: each SAE feature corresponds to a more interpretable concept than raw neurons do.

---

### Scalable Oversight and Debate

**The core problem**: as AI systems become more capable, humans lose the ability to directly evaluate the quality of their outputs (e.g., can a human verify whether a 500-page mathematical proof is correct?). Without reliable evaluation, RLHF breaks down.

**Scalable oversight approaches:**

**Debate** (Irving et al., 2018):
- Two AI agents debate the answer to a question; a human judge evaluates which argument is more convincing
- Claim: if one agent is truthful, it can always refute a lying agent's arguments — even if the human couldn't evaluate the answer directly
- Limitation: relies on the assumption that truth is "easier to defend" than lies, which doesn't always hold

**Recursive reward modeling (RRM)**: use a weaker AI to help humans evaluate a stronger AI's outputs by breaking the evaluation into smaller, more manageable pieces that the human can directly assess.

**Iterated amplification (Paul Christiano)**: 
1. Start with a human H and an AI A that approximates H
2. Amplify H using multiple copies of A as subagents: H^+ = "H with access to A"
3. Train a new A' ≈ H^+
4. Repeat: each iteration amplifies the human's capabilities while keeping values aligned

**Constitutional AI as weak scalable oversight**: by encoding principles into a constitution, you scale the human oversight capacity — a single human writing the constitution scales to an AI applying it to millions of examples.

---

## 10. Common Interview Questions

### SSM / Mamba Questions

**Q: What is the complexity of Mamba at inference vs training?**

Inference: O(N) per token step where N is the state dimension — just update the fixed-size state vector. No KV cache needed. This is the recurrent form.

Training: O(L log L) where L is sequence length — uses the convolutional form computed via FFT. Selective SSMs break pure convolutional computation (since B, C, Δ are input-dependent), so Mamba uses a custom parallel scan kernel that achieves O(L) in practice with good hardware utilization.

**Q: Why can't Mamba perfectly recall a token from 10,000 steps ago?**

The state h ∈ ℝ^N has fixed capacity. Each new input compresses into this state, overwriting old information proportional to Δ_k. Unlike attention which maintains a KV cache of all past tokens (exact recall), Mamba's state is a lossy compression of history. Whether a past token is recoverable depends on whether it was "remembered" by having a large Δ when encountered, and whether subsequent inputs didn't overwrite that slot.

**Q: What is the HiPPO matrix and why does it matter?**

HiPPO (High-order Polynomial Projection Operators) is a specific matrix A designed so that the ODE h'(t) = Ah(t) + Bx(t) corresponds to computing optimal polynomial approximations of the input history x(t). This gives the state a principled structure for memorizing history, rather than relying on random initialization. S4 proved that using HiPPO matrices enables learning on long-range dependencies that random or diagonal A matrices fail to capture.

---

### Test-Time Compute Questions

**Q: What is GRPO and how does it differ from PPO?**

PPO (Proximal Policy Optimization) requires training a separate value (critic) network to estimate state values, which adds significant memory and compute overhead.

GRPO (Group Relative Policy Optimization, used in DeepSeek-R1) eliminates the value network entirely:
1. For each prompt, sample G rollouts from the current policy
2. Score each rollout with a reward model
3. Compute a baseline as the mean reward across the G rollouts
4. Advantage for each rollout = (its reward - mean reward) / std of rewards
5. Optimize the policy using this group-relative advantage

No critic model needed. The group serves as its own baseline estimator. This halves memory requirements and works well when rollouts are cheap to generate.

**Q: When does Best-of-N outperform a larger model?**

BoN becomes competitive when:
1. The smaller model has high recall (can generate correct answers but not always consistently)
2. The reward model is accurate (reliably identifies correct answers)
3. N is large enough — BoN accuracy scales as 1-(1-p)^N where p is per-sample success rate
4. The task has verifiable answers (math, code) — reward model accuracy is higher

BoN fails when: the base model can never generate the correct answer (p ≈ 0), or when the reward model is unreliable (verifier is gamed).

**Q: What is the difference between an ORM and a PRM?**

ORM (Outcome Reward Model): assigns a single score to the entire response/answer. Trains on binary correctness labels. Simple, but can be gamed by models that produce incorrect reasoning but guess the right final answer.

PRM (Process Reward Model): assigns scores to each intermediate reasoning step. Trains on per-step correctness labels (expensive to collect). Much harder to game, better at rewarding actually correct reasoning, better for tree search (can evaluate partial traces).

---

### MoE Questions

**Q: What is expert collapse and how do you prevent it?**

Expert collapse: the router converges to sending all tokens to one or a few experts, while others receive no gradient and become useless. The model degenerates to a dense model of effective size k × (expert size) instead of N × (expert size).

Prevention:
1. Load balancing auxiliary loss: penalizes unequal expert utilization
2. Router z-loss: penalizes large router logit magnitudes to prevent premature saturation
3. Jitter noise: add N(0, σ²) to router logits during training
4. Capacity factors: cap tokens per expert, forcing the model to distribute load

**Q: What is expert parallelism and what are its challenges?**

In expert parallelism, different experts are placed on different devices (GPUs or nodes). For each MoE layer, tokens are dispatched to their assigned expert's device (all-to-all communication), processed locally, then returned (another all-to-all). 

Challenges:
- All-to-all is a bandwidth-intensive collective — scales poorly across nodes with limited interconnect
- Load imbalance: if one device's experts are overloaded, others wait (capacity factor mitigates but doesn't eliminate this)
- Routing decisions are made before tokens are sent — no ability to dynamically rebalance during a batch

---

### Long Context Questions

**Q: Explain Flash Attention's memory complexity and why it matters.**

Standard attention materializes the N×N attention score matrix in GPU HBM: O(N²) memory. For N=128K tokens and bf16, that's 128K² × 2 bytes = 32GB for attention alone, before any other activations.

Flash Attention computes attention in tiles that fit in SRAM (typically 20–40MB), maintaining only running statistics (online softmax: running max and sum for numerical stability). The full attention matrix is never written to HBM. Memory footprint: O(N) for storing Q, K, V, and outputs.

**Q: What is the lost-in-the-middle problem and why does it occur?**

LLMs show a U-shaped performance curve on long-context retrieval: performance is high for information at the beginning (primacy bias) or end (recency bias) of the context, and significantly lower for middle positions.

Mechanistically: attention has learned to focus on early tokens (attention sinks) and recent tokens (locality patterns from causal language modeling). Middle tokens receive low cumulative attention across layers. This is partly a training distribution artifact — most training sequences are short, so the model rarely had to attend to middle-document information.

---

### Synthetic Data Questions

**Q: What is model collapse and how do you mitigate it?**

Model collapse: progressive degradation of model quality when models are trained on data generated by previous model generations. Each generation amplifies the previous model's biases and suppresses the tails of the original distribution. Eventually, the model produces low-diversity, repetitive outputs.

Mitigations:
1. Always include real human data in training (maintain a fixed anchor)
2. Use diverse sampling temperatures during generation (high temperature = more tail coverage)
3. Monitor generation diversity metrics across iterations (n-gram entropy, vocabulary coverage)
4. Use verification/filtering for quality, not just memorization prevention
5. Avoid closed loops — periodically re-introduce human-generated or externally verified data

---

### PEFT Questions

**Q: Why does QLoRA work? Isn't quantization lossy?**

QLoRA works because of the separation of concerns:
1. The frozen base model in NF4 quantization retains most of the model's "knowledge" — 4-bit is lossy but the LoRA adapters are trained to compensate for the quantization noise
2. The LoRA adapters themselves are stored and computed in bf16 — no precision loss in the trainable parameters
3. NF4 is specifically designed for neural network weights (which are approximately normally distributed) — it allocates more quantization bins where weights are denser, minimizing quantization error

The resulting model is slightly worse than full bf16 fine-tuning but within 1-2% on most tasks, while fitting on 4-8× fewer GPUs.

**Q: When would you choose adapters over LoRA?**

Adapters: when you need to swap fine-tuning adapters at inference time without reloading the model (since adapters are separate modules, not merged into weights). Also useful when the target task requires injecting information at specific layer positions.

LoRA: when you want zero additional inference latency (merge BA into W after training), and when you're fine-tuning for a single task deployment. LoRA is generally preferred for production deployments.

---

### Agents Questions

**Q: What is MCP and how does it differ from function calling?**

Function calling is an application-specific mechanism: each application defines its own tools in a custom schema. Tools are not portable across applications.

MCP is a standardized open protocol (like HTTP for web services):
- Defines a universal interface for tools, resources, and prompts
- Any MCP server (e.g., a GitHub MCP server) works with any MCP client (Claude, Cursor, custom apps) without modification
- Enables a marketplace/ecosystem of reusable tools
- Supports stateful sessions (unlike stateless function calling)

**Q: What are the main failure modes of multi-agent systems?**

1. Error propagation: mistakes in early agents (wrong data, misunderstood requirements) cascade — downstream agents operate on incorrect premises with no way to detect the root cause
2. Context explosion: each subagent conversation accumulates history; at multi-hour task horizons, context windows fill and critical information is lost or truncated
3. Compound error rates: N agents each with error rate ε produce a system error rate approaching 1-(1-ε)^N — even small per-agent errors compound badly over long pipelines
4. Coordination overhead: communication between agents (tool calls, message passing) adds latency; complex coordination protocols can dominate task time
5. Safety: agents can take irreversible real-world actions; sandboxing and approval gates add friction but are necessary for high-stakes tasks

---

### Safety / Interpretability Questions

**Q: What is the superposition hypothesis?**

Neural networks with d-dimensional activation vectors can represent far more than d distinct features because features can be stored as nearly-orthogonal vectors in the same space. This is feasible when features are **sparse** (at any given input, most features are off), because the interference between simultaneously active features is then rare.

Superposition explains why individual neurons are polysemantic (respond to multiple unrelated concepts) — they encode parts of multiple superposed features. It implies that standard feature attribution methods (looking at which neurons activate) are misleading, and that interpretability requires decomposing the superposed space.

**Q: What are sparse autoencoders used for in interpretability?**

SAEs are trained to decompose model activations into a higher-dimensional, sparse feature space:
- Input: model activation a ∈ ℝ^d
- Output: sparse feature vector f ∈ ℝ^m (m >> d) where most entries are 0
- Reconstruction: â = W_dec f ≈ a

The sparsity penalty (L1 on f) forces the SAE to find features that are rarely simultaneously active. In practice, each SAE feature (column of W_dec) often corresponds to an interpretable concept (a specific name, a programming language, a logical operation).

This allows researchers to:
1. Identify what concepts a model knows about
2. Find circuits by tracking which SAE features a token activates across layers
3. Causally intervene — clamp SAE features to specific values and observe model behavior changes

---

*Last updated: May 2026 | Coverage: 2023–2025 developments | Focus: interview precision over breadth*
