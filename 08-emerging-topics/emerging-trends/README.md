---
module: Emerging Topics
topic: Emerging Trends in ML (2023–2025)
subtopic: ""
status: unread
tags: [emergingtopics, ml, emerging-trends-in-ml-2023-2025]
---
# Emerging Trends in ML (2023–2025)

> *Snapshot: June 2026. Every page in this folder is fast-moving frontier content — a current-state map, not settled canon. Re-verify model names, benchmarks, and SOTA claims before quoting them.*

## Files in This Folder

| File | What it covers |
| :--- | :--- |
| [2025-frontier-models.md](2025-frontier-models.md) | DeepSeek-V3, o3, MoE specifics, KV cache optimization |
| [frontier-ai-developments-2025.md](frontier-ai-developments-2025.md) | Deployment-focused long-context and speculative decoding |
| [state-space-models.md](state-space-models.md) | S4, Mamba, HiPPO, hybrid SSM/attention architectures |
| [large-reasoning-models.md](large-reasoning-models.md) | Test-time compute, RLVR, GRPO, PRMs, o1/R1-style reasoning |
| [mixture-of-experts.md](mixture-of-experts.md) | Top-k routing, load balancing, Switch/DeepSeek-V3 MoE |
| [multimodal-architectures.md](multimodal-architectures.md) | CLIP, BLIP-2, LLaVA, iRoPE, audio/video DiT |
| [diffusion-models.md](diffusion-models.md) | Denoising diffusion, score matching, latent diffusion |
| [agentic-ai-systems.md](agentic-ai-systems.md) | Agent architectures, tool use, multi-agent coordination |
| [advanced-rag-and-memory.md](advanced-rag-and-memory.md) | Advanced retrieval, long-term memory architectures for LLMs |
| [post-training-and-alignment.md](post-training-and-alignment.md) | RLHF, DPO, Constitutional AI, RLAIF |
| [small-language-models-and-edge.md](small-language-models-and-edge.md) | Phi, distillation, on-device/edge deployment |
| [vector-databases.md](vector-databases.md) | ANN indexing, HNSW, vector DB architecture for retrieval |
| [agi-and-asi.md](agi-and-asi.md) | AGI/ASI definitions, timelines, capability trajectories |

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

SSMs model sequences as linear dynamical systems with a fixed-size hidden state vector, achieving O(N) recurrent inference (no KV cache) and O(n log n) convolutional training via FFT. **S4** uses the HiPPO matrix (High-order Polynomial Projection Operators) for optimal history compression via Legendre polynomial basis functions. **Mamba** adds input-dependent gates (B, C, Δ) enabling selective token filtering with a hardware-aware parallel associative scan in SRAM — ~5× throughput improvement over Transformers at 2K sequence length. The key limitation: fixed-size state cannot store all key-value associations, so Mamba underperforms attention on exact associative recall. **Hybrid architectures** (Jamba: 1 attention per 7 Mamba layers; RWKV linear attention; Zamba shared attention) recover retrieval capability at lower cost.

> Full coverage — continuous-time SSM derivation, ZOH discretization, HiPPO matrix, S4 DPLR, Mamba selective scan, RWKV, Jamba hybrid: [state-space-models.md](state-space-models.md)

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

MoE replaces dense FFN sublayers with N expert FFNs and a learned top-k router: `y = Σ_{i ∈ Top-k(G(x))} G_i(x) · E_i(x)`. Total parameters scale with N; FLOPs per token scale with k. Mixtral 8x7B: 46.7B total / 12.9B active — Llama 2 70B quality at 13B inference cost. The central challenge is **routing collapse**: naive routing degrades to always using the same few experts. Solutions: auxiliary load-balancing loss `L = α·N·Σ f_i·P_i`, router z-loss (penalizes large logit magnitudes), expert capacity caps, and DeepSeek-V3's bias-based auxiliary-loss-free routing. All experts must reside in GPU memory regardless of how many are active; expert parallelism requires all-to-all communication.

> Full coverage — top-k routing derivation, load balancing, Switch/Expert Choice/Soft MoE variants, fine-grained and shared experts, DeepSeek-V3 routing: [mixture-of-experts.md](mixture-of-experts.md)

---

## 4. Long Context and Efficient Attention

Standard attention materializes an N×N matrix in GPU HBM: O(N²) memory. At N=128K tokens, ~16GB per head. **Flash Attention** (FA1/FA2/FA3) tiles Q, K, V into SRAM blocks, never materializing the full matrix — reduces memory to O(N), IO complexity O(N²/M). FA3 on H100 uses WGMMA + TMA pipelining for ~1.5–2× over FA2. **Ring Attention** distributes sequences across D devices, passing K/V in a ring while overlapping with compute — enables million-token sequences. **RoPE scaling**: YaRN applies per-dimension scale factors (high-freq dims should not be scaled; low-freq interpolated); LongRoPE uses evolutionary search for per-dimension factors. Key practical caveat: **lost-in-the-middle** — LLMs retrieve reliably from context boundaries but fail in the middle (attention sinks cause a U-shaped recall curve). Place relevant content at context boundaries in RAG pipelines.

Context window evolution: GPT-2 (2019): 1K → GPT-3 (2020): 2K → Claude 2 (2023): 100K → Gemini 1.5 Pro (2024): 1M → Llama 4 Scout (2025): 10M.

> Also see: [frontier-ai-developments-2025.md](frontier-ai-developments-2025.md) for deployment-focused long-context and speculative decoding discussion.

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

**CLIP** trains image and text encoders jointly with InfoNCE contrastive loss on 400M pairs, creating a shared embedding space enabling zero-shot classification. **BLIP-2** introduces the Q-Former — N learnable query tokens that bridge a frozen ViT encoder and frozen LLM; only the Q-Former is trained (image-text matching + captioning + VQA). **LLaVA** takes a simpler approach: project CLIP features into the LLM embedding space via a single MLP, with instruction-following data generated synthetically. Production systems (GPT-4V, Gemini, Claude) use modality-specific encoders with late fusion. **VQ-VAE** enables image tokenization via vector quantization to a learned codebook with straight-through gradients. **Early fusion** (Llama 4's iRoPE) passes image patches and text tokens through the same transformer from layer 1 — richer cross-modal interaction but requires training from scratch. Audio: log-mel spectrograms → Whisper (seq2seq, 680K hours); EnCodec/SoundStream → discrete audio tokens. Video: Sora uses a diffusion transformer on spacetime patches.

> Full coverage — ViT dynamic tiling, CLIP math, BLIP-2/Q-Former, LLaVA, iRoPE, audio/video DiT, world model hypothesis: [multimodal-architectures.md](multimodal-architectures.md)

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

**SSMs** (see [state-space-models.md](state-space-models.md) for full derivations):
- S4: HiPPO matrix A for optimal Legendre polynomial history compression. Recurrent O(N) inference, convolutional O(n log n) training.
- Mamba: input-dependent B, C, Δ — selective filtering via hardware-aware parallel associative scan in SRAM. ~5× throughput vs Transformers at 2K.
- Weakness: fixed-size state cannot store all key-value associations. Underperforms attention on associative recall. Hybrid (Jamba: 1 attention per 7 Mamba) recovers recall.

**Test-time compute** (see [large-reasoning-models.md](large-reasoning-models.md) for full coverage):
- CoT turns fixed-depth computation into adaptive-depth. Harder problems generate more reasoning tokens.
- RLVR: reward correct final answers without supervising reasoning steps. Aha-moment behaviors emerge from RL.
- GRPO: group-relative baselines replace critic network. Sample G rollouts per prompt, normalize advantages within group.
- PRMs score intermediate steps, harder to game than ORMs. PRM + BoN or beam search outperforms BoN alone.

**MoE** (see [mixture-of-experts.md](mixture-of-experts.md) for full coverage):
- Top-k routing + load balancing loss: L = α·N·Σ f_i·P_i. Too small α → collapse; too large → hurts task loss.
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
