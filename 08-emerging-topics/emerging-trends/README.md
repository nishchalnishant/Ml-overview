# Emerging Trends in ML (2023–2025)

---

## Table of Contents
1. [State Space Models (SSMs)](#1-state-space-models-ssms)
2. [Test-Time Compute Scaling](#2-test-time-compute-scaling)
3. [Mixture of Experts (MoE)](#3-mixture-of-experts-moe)
4. [Long Context and Efficient Attention](#4-long-context-and-efficient-attention)
5. [Synthetic Data and Data Flywheels](#5-synthetic-data-and-data-flywheels)
6. [Multimodal Foundation Models](#6-multimodal-foundation-models)
7. [Efficient Fine-tuning (PEFT)](#7-efficient-fine-tuning-peft)
8. [AI Agents and MCP](#8-ai-agents-and-mcp)
9. [AI Safety Frontiers](#9-ai-safety-frontiers)
10. [Key Interview Points](#10-key-interview-points)

---

## 1. State Space Models (SSMs)

### The Problem

Transformers have O(n²) attention cost and an O(n) KV cache. At 100K tokens, the KV cache alone consumes tens of gigabytes. At inference time, generating token t+1 requires reading the full cached K,V history — every prior token contributes to every subsequent token. The cost is real and the memory is not free.

This matters most for two use cases: very long sequences (genomics, audio, long documents) and streaming inference on edge hardware where the KV cache cannot fit.

---

### The Core Insight: Sequences as Linear Dynamical Systems

Every sequence model must answer: "given everything I've seen, what should I predict next?" Transformers answer by recomputing attention over the full history. An alternative is to compress the full history into a fixed-size state vector and update it incrementally — the same approach used in control theory and signal processing for decades.

A continuous-time linear dynamical system:

```
h'(t) = A h(t) + B x(t)    # state update
y(t)  = C h(t) + D x(t)    # output
```

`h(t)` is the hidden state — a fixed-size summary of history. Inference is O(N) per step regardless of sequence length. Training uses the convolutional representation: unroll the recurrence into a convolution kernel and compute via FFT in O(n log n).

The question is: which matrix A preserves long-range history without vanishing or exploding?

---

### S4: Structured State Space Sequence Model

S4 (Gu et al., 2021) answers the matrix A question by using the **HiPPO matrix** (High-order Polynomial Projection Operators). HiPPO-LegS projects incoming signal onto Legendre polynomials, which are designed to memorize history optimally — each basis function tracks a different timescale.

Discretized with step size Δ (zero-order hold):

```
Ā = exp(ΔA)
B̄ = (ΔA)^{-1}(exp(ΔA) - I) · ΔB

h_k = Ā h_{k-1} + B̄ x_k
y_k = C h_k + D x_k
```

Two computation modes:
- **Recurrent (inference):** O(N) per step, fixed memory — no KV cache
- **Convolutional (training):** compute kernel K = (CB̄, CĀB̄, CĀ²B̄, ...) as one FFT convolution, O(n log n)

**What breaks:** A is a fixed global matrix — it does not depend on input content. S4 cannot do content-based retrieval: it cannot "look back at" a specific earlier token because the state is a lossy compression, not a full cache. In the associative recall task (retrieve value for a specific key seen earlier), S4 underperforms attention substantially.

---

### Mamba: Selective State Spaces

Mamba (Gu & Dao, 2023) makes B, C, and Δ **input-dependent**:

```
B_k = s_B(x_k)
C_k = s_C(x_k)
Δ_k = softplus(s_Δ(x_k))
```

Now the model selectively filters: small Δ means the state barely updates (input is ignored); large Δ means the input strongly updates the state. The model can selectively "forget" irrelevant tokens and "focus" on relevant ones — closing the gap with attention on recall tasks.

Since B, C, Δ now vary per token, the training cannot be purely convolutional. Mamba implements a **hardware-aware parallel associative scan** using a custom CUDA kernel that:
1. Keeps the scan in SRAM (on-chip), not HBM (off-chip)
2. Uses parallel prefix scan to compute the recurrence across positions
3. Avoids materializing the full state sequence in HBM

Result: ~5× better throughput than Transformers at sequence length 2K; gap widens with longer sequences.

```
Mamba SSM block:
Input x
→ Linear projection (expand 2×)
→ Split into z (gate) and x' (content)
→ x': 1D depthwise conv → SiLU → selective SSM scan
→ Gate: x' * sigmoid(z)
→ Linear projection back to d_model
```

**What breaks:** Mamba's fixed-size state still cannot perfectly store all prior key-value associations. In exact associative recall benchmarks, Mamba underperforms full attention. It is not a universal replacement for attention — it is a different tradeoff: better throughput and recurrent inference, weaker at precise discrete retrieval.

---

### Hybrid Architectures: Jamba and Zamba

The field converged on a practical answer: **hybridize**. Most sequence modeling is cheap compression — Mamba handles that. Precise retrieval needs a few attention layers to catch it.

**Jamba** (AI21, 2024): ~1 attention layer per 7 Mamba layers, combined with MoE. 52B total / 12B active, 256K context on a single 80GB A100. Attention layers handle retrieval; Mamba layers handle cheap long-range compression.

**Zamba** (Zyphra, 2024): a single shared attention block reused at multiple points in the network — sparse attention coverage sufficient to recover recall performance at lower parameter cost than dedicated attention layers.

The lesson: attention's O(n²) cost is wasted on most of the sequence modeling problem. Mamba-style recurrence handles the bulk cheaply. The minimum viable amount of attention is enough to preserve retrieval capability.

---

## 2. Test-Time Compute Scaling

### The Problem

The dominant ML scaling narrative: more training compute → better model. But training compute is slow and expensive to acquire. The question that emerged in 2024: can you instead use more compute at inference time — after training — to get better answers on hard problems?

The concrete failure case that motivated this: a model answers a hard math problem incorrectly in one forward pass. But if you ask it to reason step-by-step, it gets it right. The information needed to solve the problem is in the model's weights — the model just needed more computation to extract it.

---

### The Core Insight: Inference Compute as Adaptive Computation Depth

A standard transformer forward pass has fixed depth: d_model × n_layers × seq_len operations. Chain-of-thought (CoT) transforms this into variable-depth computation: each reasoning step requires a new forward pass. A harder problem generates more reasoning tokens and therefore more total computation.

This is not a prompt engineering trick. It is a fundamental change in the computation graph: from fixed-depth to adaptive-depth, where the model allocates compute according to problem difficulty.

The key requirement: **verifiable rewards**. If you can check whether the final answer is correct (math: ground truth; code: running tests), you can train the model to discover whatever reasoning strategy leads to correct answers — without supervising the intermediate steps.

---

### RLVR: Reinforcement Learning with Verifiable Rewards

Training loop:
1. Sample problem p from training set
2. Generate G rollouts (full reasoning traces + final answers) from current policy
3. Score each rollout: +1 if final answer is correct, 0 otherwise
4. Update policy to increase probability of high-reward reasoning traces

This produces the "aha moment" behaviors that characterize reasoning models: backtracking ("Wait, let me reconsider..."), self-checking, extended deliberation. These emerge from RL, not from explicit supervision.

**DeepSeek-R1** training recipe:
1. Cold start: SFT on small set of human-written long chain-of-thought examples (establishes format)
2. GRPO (Group Relative Policy Optimization): sample G rollouts per prompt, normalize advantages within the group — no separate value network required
3. Rejection sampling + SFT: collect high-quality reasoning traces, fine-tune for stability
4. Final RLHF for helpfulness

GRPO replaces the separate critic network in PPO with group-relative baselines:

```
For prompt p, sample rollouts {o_1, ..., o_G}
Score: {r_1, ..., r_G}
Advantage: A_i = (r_i - mean(r)) / std(r)
Update via clipped PPO objective on A_i
```

Halves memory requirements vs PPO.

---

### Best-of-N and Process Reward Models

**Best-of-N (BoN):** generate N independent completions, score each with a reward model, return the highest-scoring one. Empirically, BoN accuracy scales consistently with N on math benchmarks up to N ≈ 1000. The generator does not change — only inference budget increases.

**Outcome Reward Models (ORMs):** score only the final answer. Fast to train (binary correct/incorrect), but vulnerable to shortcuts — a model that reaches the right answer via wrong reasoning gets rewarded.

**Process Reward Models (PRMs):** score each intermediate step. Harder to game — wrong reasoning cannot produce a correct step score. Requires step-level labels (OpenAI's PRM800K: 800K step labels). PRMs enable step-level beam search: prune reasoning trees by intermediate quality, not just final answer.

**Monte Carlo Tree Search for reasoning:**

```
UCB score = Q(s,a) + c · sqrt(ln N(s) / N(s,a))
```

Build a tree of reasoning prefixes. Expand high-UCB nodes with LLM-generated next steps. Rollout to final answer. Backpropagate correctness. Expensive but finds high-quality reasoning paths.

**What breaks:** PRMs require human step-level labels, which are expensive and hard to scale. BoN requires N full forward passes. At large N, test-time compute becomes more expensive per query than serving a larger model. The tradeoff between model size and inference compute depends on the task and hardware — not universally resolved.

---

## 3. Mixture of Experts (MoE)

### The Problem

Dense neural networks apply every parameter to every input token. A 70B dense model applies all 70B parameters to every token in every forward pass. This couples model capacity (parameters = knowledge) directly to compute (every parameter runs on every input).

The question: can you add more parameters — more capacity, more knowledge — without paying proportional compute cost?

---

### The Core Insight: Conditional Computation

Replace the dense FFN sublayers with a bank of N expert FFNs. A learned router selects k of them per token. Total parameters scale with N; FLOPs per token scale with k.

```
MoE layer output:
y = Σ_{i ∈ Top-k(G(x))} G_i(x) · E_i(x)

G(x) = softmax(W_g · x)     # router logits
E_i(x) = FFN_i(x)           # i-th expert FFN
Top-k selects k largest gate values
```

Since FFN layers are ~2/3 of transformer parameters, replacing them with MoE layers gives a parameter multiplier without proportional compute increase. Mixtral 8x7B: 46.7B total parameters, 12.9B active per token. Llama 2 70B quality at Llama 2 13B inference cost.

---

### Load Balancing: The Central Challenge

Naive top-k routing leads to **routing collapse**: the router learns to always use the same few experts, which receive all gradients and improve while others stagnate.

**Load balancing loss** (auxiliary, added to task loss):

```
L_balance = α · N · Σ_i f_i · P_i

f_i = fraction of tokens routed to expert i in the batch
P_i = mean gate probability for expert i
α   = balancing coefficient (0.01–0.1)
```

Encourages uniform expert utilization. Tuning α is critical: too small → collapse; too large → dominates task loss and hurts model quality.

**Router z-loss** (ST-MoE, Zoph et al. 2022):

```
L_z = β · (1/B) Σ_x (log Σ_i exp(h_i(x)))²
```

Penalizes large router logit magnitudes. Stabilizes training by preventing the router from becoming too confident early.

**Expert capacity:** set a capacity cap C tokens per expert per batch. Tokens routed to an over-capacity expert are dropped (or fall through a residual connection). Prevents one expert from receiving all tokens in a batch but introduces information loss.

**Monitoring routing entropy:**

```python
routing_entropy = -sum(p_i * log(p_i) for p_i in expert_utilization)
# Target: close to log(N) (maximum entropy = uniform)
```

---

### Architecture Variants

**Switch Transformer** (Fedus et al., 2021): top-1 routing (k=1). Simpler than k=2 and works if: router initialized with small weights, router kept in float32 during mixed-precision training, load balancing loss applied. Scaled to 1.6T parameters; matched T5-11B at 7× fewer FLOPs.

**Expert Choice routing** (Zhou et al., 2022): each expert selects its top-k tokens rather than each token selecting experts. Guarantees perfect load balance. Breaks autoregressive inference (experts see future tokens in the batch).

**Hash routing:** deterministic assignment by token hash. Zero routing overhead, perfect balance, no content-awareness. Performs surprisingly well for some tasks.

**What breaks:** All MoE experts must reside in GPU memory simultaneously even if only k are active. "22B active params" does not mean "22B params in memory" — all 235B must be loaded. Expert parallelism (sharding experts across devices) requires all-to-all collective communication, which can become the bottleneck on large clusters. MoE fine-tuning is fragile: expert collapse under small datasets, routing instability with aggressive learning rates.

---

## 4. Long Context and Efficient Attention

### The Problem

Context window evolution:

| Year | Model | Context |
|------|-------|---------|
| 2019 | GPT-2 | 1,024 |
| 2020 | GPT-3 | 2,048 |
| 2023 | Claude 2 | 100,000 |
| 2024 | Gemini 1.5 Pro | 1,000,000 |
| 2025 | Llama 4 Scout | 10,000,000 |

Standard attention materializes an N×N matrix in GPU HBM (high-bandwidth memory): O(N²) memory, O(N²) compute. At N=128K tokens, the attention matrix alone is 16GB per head. This is not an engineering bottleneck to optimize around — it is a mathematical constraint that requires a different algorithm.

---

### Flash Attention: IO-Aware Tiling

Flash Attention (Dao et al., 2022) does not reduce theoretical FLOP count — it reduces **memory IO** between HBM and SRAM. GPUs are memory-bandwidth limited, not compute limited, for attention.

Algorithm:
1. Tile Q, K, V into blocks that fit in SRAM (on-chip fast memory)
2. Compute attention block-by-block, maintaining running softmax statistics (online safe softmax: running max and sum for numerical stability)
3. Never materialize the full N×N matrix in HBM
4. Fuse backward pass to avoid re-reading Q, K, V

```
Memory: O(N) vs O(N²)
IO: O(N²/M) where M = SRAM size
```

**Flash Attention 2** (2023): parallelizes across sequence dimension (not just batch/head), better work partitioning across GPU warps. ~2× speedup over FA1, ~70% of theoretical GPU FLOP utilization.

**Flash Attention 3** (2024, H100/Hopper): exploits WGMMA (Warpgroup Matrix Multiply Accumulate) instructions. Overlaps softmax and GEMM via pipelining — while computing attention scores, simultaneously load next K/V tile via TMA (Tensor Memory Accelerator). ~1.5–2× over FA2 on H100.

---

### Ring Attention: Sequences Across Devices

When the sequence does not fit on a single GPU at all, Ring Attention (Liu et al., 2023) distributes it across D devices:

1. Each device holds N/D tokens of Q, K, V
2. Each device computes attention for its Q chunk against its local K/V chunk
3. Devices pass K/V chunks to the next device in a ring, overlapping communication with computation
4. After D-1 passes, each device has seen all K/V and can produce its final output

Communication (K/V transfer around ring) overlaps with compute (attention on current K/V chunk). Memory per device: O(N/D). Enables million-token sequences across many GPUs.

---

### RoPE Scaling: YaRN and LongRoPE

Most modern LLMs use **RoPE** (Rotary Position Embeddings). Position m is encoded by rotating Q and K vectors:

```
For dimension pair (2i, 2i+1), position m:
[q_{2i}cos(mθ_i) - q_{2i+1}sin(mθ_i), ...]
where θ_i = 10000^{-2i/d}
```

A model trained with max length L has only seen rotation angles for positions 0...L-1. Position m > L produces out-of-distribution angles; performance degrades sharply.

**Simple interpolation:** map position m to m × (L/L_new) — compress all positions into the seen range. Works but reduces positional resolution. Fine-tuning required.

**YaRN** (Peng et al., 2023): different RoPE frequency dimensions need different scaling. High-frequency dimensions (small θ) should not be scaled — they are dense and scaling hurts. Low-frequency dimensions (large θ) should be interpolated. Mid-frequency: smooth ramp between strategies. Also scales attention temperature by √(log(L_new)/log(L)) to prevent attention entropy collapse at long context.

```python
def yarn_scale(dim_idx, d_model, base=10000, scale=8):
    theta = base ** (-2 * dim_idx / d_model)
    if theta > 4:    # high freq: no scaling
        return 1.0
    elif theta < 1:  # low freq: full interpolation
        return scale
    else:            # smooth ramp
        return 1 + (scale - 1) * (theta - 4) / (1 - 4)
```

**LongRoPE** (2024): searches for per-dimension scaling factors via evolutionary optimization. Claims 2M token context with minimal perplexity degradation.

---

### Lost-in-the-Middle: Long Context Does Not Mean Long Retrieval

Liu et al. (2023): LLMs retrieve information reliably from the beginning and end of long contexts but fail in the **middle**. Performance degrades with a U-shaped curve — primacy and recency effects dominate.

Cause: attention sinks (Xiao et al., 2023) — certain tokens ([BOS], punctuation) attract disproportionate attention. Middle tokens receive lower cumulative attention weights.

Practical implications:
- In RAG pipelines, order retrieved chunks to place most relevant content at context boundaries, not middle
- A 1M token context window does not mean 1M token retrieval. Test retrieval accuracy by position before relying on it
- RAG vs long context is not a binary choice: use retrieval to find relevant chunks, then context for holistic reasoning

**What breaks:** Longer context windows increase KV cache memory requirements proportionally. Serving a 1M-token query is 1000× the memory cost of a 1K-token query. Long context is a capability, not a free lunch.

---

## 5. Synthetic Data and Data Flywheels

### The Problem

Internet-scale pretraining consumed available high-quality text data. Estimates suggest the "unique quality tokens" ceiling will be reached within a few years at current scaling rates. What do you train on when you've trained on everything?

The same scarcity applies to fine-tuning: human-annotated reasoning traces for hard math and code cost $1–10 per example. Getting millions of long chain-of-thought traces this way is prohibitive.

---

### The Core Insight: Models Can Supervise Themselves

If you can verify correctness without human judgment — ground-truth answers for math, test suites for code — then you don't need human annotation. Generate many candidate solutions, keep the correct ones, fine-tune on those. A better model generates better training data, which trains an even better model.

This is the **data flywheel**: capability improvement is self-reinforcing when verifiable tasks provide the signal.

---

### Rejection Sampling Fine-tuning (RFT)

```
1. Start with model M_0
2. For each problem p:
   a. Sample N completions: {s_1, ..., s_N} ~ M_0(p)
   b. Verify: keep only correct solutions
3. Fine-tune M_0 on correct solutions → M_1
4. Repeat with M_1 as generator
```

Each iteration produces a model that solves more problems, generates more correct solutions for harder problems, and produces better training data for the next iteration. The key constraint: you must have a reliable verifier. For math: check final numeric answer. For code: run test suite.

---

### Constitutional AI (CAI) and RLAIF

CAI (Bai et al., Anthropic 2022) addresses a different bottleneck: getting harmlessness labels. Human labeling for "is this response harmful?" is expensive and inconsistent. AI feedback can scale better.

**Phase 1 — SL-CAI:**
1. Sample potentially harmful responses from initial model
2. Ask the same model to revise its response according to a **constitution** (list of principles)
3. Fine-tune on the revised responses

**Phase 2 — RLAIF:**
1. Generate response pairs
2. Use a separate feedback model with the constitution to pick which is better
3. Train a preference model on AI-generated labels
4. Run PPO against this preference model — same as RLHF but AI labeler, not human

The risk: feedback model inherits generator biases. The system optimizes against a biased evaluator, reinforcing its own blind spots.

---

### Phi Models: Textbook Quality Over Web Scale

Microsoft Phi series demonstrated that **data quality matters more than quantity at smaller scales**.

- **Phi-1** (1.3B): trained on synthetically generated "textbook quality" code examples. Outperformed models 10× larger on HumanEval.
- **Phi-3-mini** (3.8B): competitive with Llama 3 70B on many benchmarks using aggressive synthetic data curriculum.

Synthetic data recipe:
1. Generate didactic content (worked examples, step-by-step explanations) using a large teacher model
2. Mix with filtered web data (educational quality classifier on Common Crawl)
3. Deduplicate aggressively
4. Curriculum ordering: simple → complex

The underlying mechanism: models trained on structured, pedagogical text learn to reason step-by-step. Random internet text lacks this structure.

---

### Model Collapse

Training on AI-generated data creates a distribution shift problem:

```
Human data: full distribution p(x)
M_0 approximates p(x), generating q_0(x) — tails trimmed
M_1 trained on q_0(x) approximates q_0(x) — further trimmed
...
M_n: increasingly narrow, low-diversity distribution
```

Low-probability events (creative, unusual, diverse outputs) are suppressed each generation. The model converges toward the mode of the distribution.

**Mitigations:**
- Always mix real human data (never train purely on synthetic)
- Use diverse sampling (high temperature, varied prompts)
- Monitor vocabulary diversity and n-gram entropy across generations
- Verify for quality not just correctness — over-filtering amplifies collapse

**What breaks:** RLAIF and closed-loop synthetic data systems also suffer from **echo chamber** failure: if the feedback model and generator share the same base model and training pipeline, AI feedback reinforces shared biases rather than correcting them.

---

## 6. Multimodal Foundation Models

### The Problem

Language models process tokens. Images, audio, and video are not naturally tokens. The question is how to bridge modalities: do you train separate specialist models and fuse them late, or do you train a single model on all modalities from the start?

Each choice has a different cost and a different capability profile.

---

### CLIP: Shared Embedding Space

CLIP (Radford et al., 2021) trains image and text encoders jointly via contrastive loss on 400M image-text pairs:

```
L = -log(exp(sim(I_i, T_i)/τ) / Σ_j exp(sim(I_i, T_j)/τ))
```

Maximize cosine similarity of matching (image, text) pairs; minimize for non-matching. Creates a shared embedding space where semantically related images and text are nearby.

Zero-shot classification: compute image embedding similarity against text embeddings of all class descriptions ("a photo of a dog", "a photo of a cat"). No fine-tuning needed.

CLIP's limitation: it produces image representations aligned with text, but it is not a generative model and cannot reason about spatial relationships or fine-grained visual details.

---

### BLIP-2 and LLaVA: Connecting Vision to Language Models

**BLIP-2** (Li et al., 2023): introduces the **Q-Former** — a lightweight transformer that bridges a frozen image encoder (ViT) and a frozen LLM. N learnable "query" tokens attend to image features and extract relevant information for language generation. Only the Q-Former is trained. Pre-training on image-text matching, captioning, and VQA; then connected to LLM.

**LLaVA** (Liu et al., 2023): simpler. Project CLIP ViT features into the LLM embedding space via a single linear layer or MLP. Generate instruction-following training data synthetically using GPT-4 (write conversations about images from COCO captions). Fine-tune LLaMA on this data. Surprisingly competitive with BLIP-2 at much lower cost.

**The production approach (GPT-4V, Gemini, Claude):** modality-specific encoders with late fusion into the LLM. No single public architectural disclosure. Key capabilities beyond LLaVA: fine-grained OCR, spatial reasoning, multi-image comparisons, diagram understanding.

---

### Image Tokenization: VQVAE and dVAE

To process images as tokens in a transformer, images need discrete token IDs.

**VQVAE** (van den Oord et al., 2017):
1. Encoder maps image to latent grid z_e(x)
2. Vector quantization: each spatial position is mapped to nearest code in a learned codebook C of K entries:
   ```
   z_q(x) = argmin_{c_k ∈ C} ||z_e(x) - c_k||_2
   ```
3. Decoder reconstructs image from quantized latents
4. Straight-through estimator passes gradients through the discrete argmin

Result: image → sequence of discrete tokens from a vocabulary of size K.

**dVAE** (DALL-E 1): variational formulation using Gumbel-softmax for soft categorical sampling, annealed toward hard discrete. Scales to larger codebooks.

**Why it matters:** once images are tokens, you can train autoregressive or masked models on image + text token sequences jointly. Modern image generation (DALL-E 3, Stable Diffusion) uses diffusion over continuous latents rather than discrete tokens, but the encoder-decoder structure persists.

---

### Audio and Video

**Audio:** raw waveforms → log-mel spectrograms → 2D feature maps → processed like images. **Whisper** (Radford et al., 2022): seq2seq on spectrograms, trained on 680K hours with automatic supervision. **EnCodec/SoundStream:** neural audio codecs → discrete audio tokens → LLMs model audio autoregressively.

**Video:** spatial + temporal dimensions. Challenges: 30fps × 1080p = enormous data. Sparse temporal sampling (1fps), or factorized attention (spatial within frames, temporal across frames). **Sora** (OpenAI, 2024): diffusion transformer on continuous spacetime patches, coherent minute-long video generation.

**Unified vs modality-specific:** Unified backbones (Perceiver, Flamingo, Gemini) share representations across modalities but are harder to train. Modality-specific encoders + fusion (LLaVA, BLIP-2, Whisper + LLM) leverage existing checkpoints but limit deep cross-modal interaction. Most production deployments use specialist encoders with late fusion.

**What breaks:** Llama 4's early fusion (image patches + text tokens through the same transformer from layer 1) requires training from scratch — cannot start from a text-only pretrained model. This dramatically increases training cost. The architecture advantage (richer cross-modal interaction) is real but the infrastructure requirement is high.

---

## 7. Efficient Fine-tuning (PEFT)

### The Problem

Full fine-tuning of a 70B model:
- Weights: 70B × 2 bytes (bf16) = 140GB
- Gradients: 140GB
- Adam states (momentum + variance): 560GB
- Total: ~840GB minimum — 10+ A100 80GB GPUs just for optimizer state

Most practitioners do not have 10 A100s. Even those who do want to maintain multiple fine-tuned versions without storing full model copies.

---

### The Core Insight: Most Parameters Do Not Need to Change

Fine-tuning adjusts a pre-trained model to a new task. The pre-trained model already contains most of the knowledge needed. The adjustment is a small perturbation in a high-dimensional space. This perturbation has low intrinsic dimensionality — it can be expressed as a low-rank update.

---

### LoRA and Its Variants

**LoRA** (Hu et al., 2021): for weight matrix W ∈ ℝ^{d×k}, train two low-rank matrices instead of updating W:

```
W' = W + ΔW = W + B · A

A ∈ ℝ^{r×k}, B ∈ ℝ^{d×r}, r << min(d,k)
A ~ N(0, σ²), B = 0 at initialization (ΔW=0 initially)
```

Scaling factor α/r applied to ΔW for stability. Trainable params: r(d+k) vs d×k. At r=16, d=k=4096: 131K vs 16.7M.

At inference: merge W + BA back into W — **zero additional latency**.

**LoRA+** (2024): use different learning rates for A (input matrix) and B (output matrix). B should have a higher LR due to asymmetric gradient flow through the product BA. Meaningful improvement with no architecture change.

**DoRA** (2024): decompose the update into magnitude and direction separately:

```
W' = m · (W + BA) / ||W + BA||_c

m = ||W||_c (learnable scalar per column)
||·||_c = column-wise norm
```

Training magnitude and direction independently more closely mimics full fine-tuning update patterns. Better performance in low-rank regimes (r=4, r=8).

**QLoRA** (Dettmers et al., 2023): combine LoRA with 4-bit quantization of the frozen base model:
1. Quantize base model to NF4 (NormalFloat4) — data type optimized for normally distributed weights
2. Double quantization: quantize the quantization constants too
3. LoRA adapters in bf16 on top of 4-bit base
4. Paged optimizers for GPU memory spike management

Fine-tune a 65B model on a single 48GB GPU. Quality degradation vs full bf16: <1% on benchmarks.

---

### Adapters, Prefix Tuning, Prompt Tuning

**Adapters** (Houlsby et al., 2019): insert small bottleneck networks inside each transformer layer:

```
Adapter(x) = x + W_up · ReLU(W_down · LN(x))
W_down ∈ ℝ^{d×r}, W_up ∈ ℝ^{r×d}
```

Cannot be merged like LoRA — adds inference latency (~2–4ms per layer).

**Prefix tuning** (Li & Liang, 2021): prepend K learnable "virtual tokens" to keys and values at each attention layer:

```
Attention(Q, [P_K; K], [P_V; V])
P_K, P_V ∈ ℝ^{K×d} are learned prefix matrices
```

No inference latency (prefix extends KV, not model). Works well for generation tasks; less stable than LoRA for classification.

**Prompt tuning** (Lester et al., 2021): add learnable tokens only at the input layer, not at every attention layer. Fewer parameters than prefix tuning. Performance matches full fine-tuning only at very large model scales (>10B).

**What breaks:** LoRA requires knowing which weight matrices to adapt (typically Q, K, V projections; sometimes FFN). Rank selection (r=4 to r=64) requires hyperparameter search. For very different tasks from pretraining (e.g., domain-specific classification from a general LLM), low-rank updates may not capture the required update magnitude — full fine-tuning still wins on large distribution shift. LoRA + MoE is unstable: expert routing changes during fine-tuning can cause expert collapse.

---

## 8. AI Agents and MCP

### The Problem

Language models produce text. Real tasks require acting on the world: calling APIs, running code, reading files, browsing the web, coordinating with other models. "Acting" requires more than generating text — it requires a protocol for what an action looks like, how results come back, and how multiple tools and models coordinate.

---

### The Core Insight: Separate Tool Protocol from Model Capability

A model that can call functions is a different architecture from a model that generates text. The key insight is to standardize the interface: a model declares what tools it needs, a runtime handles tool calls, results are injected back into context. This decouples the model from the execution environment.

---

### Function Calling vs MCP

**OpenAI function calling format:**

```json
{
  "type": "function",
  "function": {
    "name": "get_weather",
    "description": "Get current weather for a location",
    "parameters": {
      "type": "object",
      "properties": {
        "location": {"type": "string"},
        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
      },
      "required": ["location"]
    }
  }
}
```

The model generates a JSON tool call, the runtime executes it, the result is returned in the next message.

**Model Context Protocol (MCP):** standardized protocol for LLM-tool interaction. Rather than each vendor defining their own function-calling format, MCP defines a universal interface. An MCP server exposes tools; an MCP client (the LLM runtime) calls them. Tools, resources, and prompts are first-class MCP constructs. Enables plug-and-play tool composition across different model providers.

---

### Computer Use and Agentic Capabilities

**Computer use** (Anthropic, 2024): model takes screenshots, identifies UI elements, generates mouse/keyboard actions. The action space is raw computer interaction rather than structured API calls. Enables agents to operate any application without API integration.

**ReAct loop:**

```
Thought: I need to check the current stock price
Action: search("AAPL stock price today")
Observation: Apple Inc (AAPL): $212.45, +1.2%
Thought: Now I can answer the user's question
Answer: The current AAPL price is $212.45
```

Each thought-action-observation cycle is a forward pass. Harder problems require more cycles — again, adaptive computation depth at inference time.

---

### Multi-Agent Architectures

**Orchestrator-subagent:** one agent plans and delegates; specialized subagents execute. Orchestrator maintains state and handles errors; subagents are stateless workers.

**Peer agents:** multiple agents work in parallel, share a message bus, coordinate via asynchronous messages. Useful when subtasks are genuinely independent (research + writing + fact-checking).

**Memory types in agent systems:**
- **In-context:** current conversation window (volatile, limited)
- **External (vector store):** retrieved by embedding similarity (semantic memory)
- **Episodic (database):** structured logs of past interactions (explicit history)
- **Parametric:** baked into model weights during training (cannot be updated at runtime)

**What breaks:** Long agentic loops accumulate errors — each tool call is a potential failure point. Agents fail to recognize when they are stuck in loops. Tool reliability matters more than in single-turn inference because errors compound over many steps. Context window management is critical: a 200K context fills quickly in a long tool-using session. Multi-agent coordination introduces new failure modes: deadlocks, message loss, inconsistent state across agents.

---

## 9. AI Safety Frontiers

### The Problem

A language model trained to predict next tokens on internet text will predict whatever tokens are most statistically likely — including instructions for harmful activities, confident-sounding misinformation, and manipulative content. Capability and harmlessness are not naturally aligned. This is the core alignment problem: how do you make a capable model also do what you want, rather than what it learned to predict?

---

### Constitutional AI (CAI) and RLAIF

Covered in Section 5 from the data perspective; the safety perspective:

The constitution is a list of principles the model should follow: be honest, do not assist with harmful activities, be respectful of human autonomy. The key property of CAI: the model critiques and revises its own outputs against the constitution without requiring human judgment on each example. This scales safety training beyond what human annotation can provide.

The fundamental limitation: the constitution is written by humans and can be incomplete or inconsistent. The model optimizes for the constitution, not for safety itself. Adversarial prompting can find corners of behavior not covered by any principle.

---

### Mechanistic Interpretability

The goal: understand what specific computations a neural network implements — not just "what does it output" but "which neurons, heads, and circuits produce that output."

**Superposition hypothesis** (Elhage et al., 2022): a neural network has more useful "features" to represent than it has neurons. It encodes multiple features in superposition using the same neurons, relying on the fact that most features are rarely active simultaneously. Sparse linear combinations of neurons represent distinct concepts.

This explains why individual neurons are difficult to interpret: most neurons are polysemantic — they participate in multiple features depending on context.

**Sparse Autoencoders (SAEs):** train an autoencoder with a sparsity penalty to find a larger set of monosemantic features:

```
z = ReLU(W_enc · (x - b_dec))        # sparse feature activations
x̂ = W_dec · z + b_dec                # reconstruction
L = ||x - x̂||² + λ||z||_1           # reconstruction + sparsity
```

The SAE's hidden units should correspond to interpretable features — things like "Python code", "emotional content", "mentions of specific entities". The larger hidden space (e.g., 4096 SAE features for a 512-dim residual stream) provides room to separate superposed features.

**Circuits:** specific paths through a model that implement specific computations. The "indirect object identification" circuit in GPT-2 (Wang et al., 2022) was traced to specific attention heads implementing copy-suppression and name-mover operations.

**What breaks:** current mechanistic interpretability methods scale to small models (GPT-2, Llama-7B) and specific narrow behaviors. Scaling to frontier models and open-ended behavior remains unsolved. SAEs produce thousands of features; manually interpreting them all is impossible.

---

### Scalable Oversight

The problem: as models become more capable, humans can no longer reliably evaluate whether model outputs are correct. A human cannot verify a 10,000-line proof or a complex codebase patch. If you cannot evaluate correctness, you cannot train on it reliably.

**Debate** (Irving et al., 2018): two models argue opposing positions; a human evaluates which argument is more compelling. The hypothesis: even if a human cannot independently verify a complex claim, they can judge which of two arguments is better. Amplifies human judgment by forcing the model to justify its position against an adversary.

**Scalable oversight via decomposition:** break complex tasks into subtasks small enough for human evaluation. Recursive decomposition: verify each step of reasoning rather than the final answer. This is the intuition behind PRMs in Section 2.

**Constitutional supervision:** use a more capable model to evaluate a less capable model's outputs. The risk: if the evaluator model has the same failure modes as the model being evaluated, the oversight provides no signal.

**What breaks:** Debate assumes humans can identify better arguments. Sophisticated models may produce more persuasive-sounding wrong arguments. The scalable oversight problem has no clean solution — it is an active research area with several promising directions but no definitive answer.

---

## 10. Key Interview Points

**SSMs:**
- S4: linear dynamical system with HiPPO matrix A for optimal history compression. Two modes: recurrent O(N) inference, convolutional O(n log n) training.
- Mamba: input-dependent B, C, Δ — model selectively filters which tokens update the state. Hardware-aware parallel associative scan in SRAM.
- Mamba weakness: fixed-size state cannot store all key-value associations. Underperforms attention on associative recall. Hybrid (Jamba: 1 attention per 7 Mamba layers) recovers recall capability.

**Test-time compute:**
- CoT turns fixed-depth computation into adaptive-depth. Harder problems generate more reasoning tokens.
- RLVR: reward correct final answers without supervising reasoning steps. Aha-moment behaviors (backtracking, self-verification) emerge from RL.
- GRPO: group-relative baselines replace critic network. Sample G rollouts per prompt, normalize advantages within group.
- PRMs score intermediate steps, harder to game than ORMs. PRM + BoN or beam search outperforms BoN alone.

**MoE:**
- Top-k routing + load balancing loss. Load balancing loss = α · N · Σ f_i · P_i. Too small α → collapse; too large → hurts task loss.
- Active params ≠ memory used. All 235B Qwen3 params must be loaded to serve 22B active.
- Expert collapse: routing entropy monitoring, z-loss, expert dropout.

**Long context:**
- Flash Attention: tile Q/K/V to SRAM, never materialize N×N in HBM. O(N) memory vs O(N²). IO complexity O(N²/M).
- Ring Attention: distribute sequence across D devices. Pass K/V in a ring, overlapping communication with compute.
- YaRN: high-frequency RoPE dimensions should not be scaled; low-frequency should be interpolated. Smooth ramp between strategies.
- Lost-in-the-middle: attention concentrates at context boundaries. Relevant content should be placed at beginning or end of context, not middle.

**Synthetic data:**
- RFT data flywheel: generate N solutions, keep correct, fine-tune, repeat.
- Model collapse: each generation trims distribution tails. Always mix real data; monitor diversity.
- Phi: textbook-quality synthetic data outperforms raw web data at small scale.

**PEFT:**
- LoRA: W' = W + BA, rank r. Merged at inference — zero latency.
- QLoRA: 4-bit NF4 base + bf16 LoRA adapters. 65B on 48GB GPU.
- DoRA: decomposes update into magnitude and direction for better low-rank fidelity.
- Adapters: cannot be merged — add inference latency. Prefix tuning: extends KV at every layer.

**Agents:**
- MCP: universal tool protocol. Model generates tool calls; runtime executes; results returned to context.
- ReAct: thought → action → observation loop. Each step is one forward pass.
- Multi-agent failures: error accumulation, context window overflow, coordination deadlocks.

**Safety:**
- Superposition: one neuron participates in multiple features. SAEs find monosemantic features in a larger sparse space via L1 penalty.
- Debate: two models argue; human judges arguments. Amplifies human oversight by forcing adversarial justification.
- Scalable oversight: verify complex tasks by decomposing into human-sized subtasks. No clean solution yet.
