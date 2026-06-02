---
module: Emerging Topics
topic: Emerging Trends
subtopic: Frontier Models
status: unread
tags: [emergingtopics, ml, emerging-trends-2025-frontier-]
---
# 2025 Frontier Models — Field Guide

> **Companion page:** [Frontier AI Developments 2025](frontier-ai-developments-2025.md) covers the same era from a trends lens (multimodal native fusion, long context, agents). Read both if you want model-by-model depth *and* cross-cutting themes without overlap fatigue.

---

## 1. The Problem

In 2023, the frontier model landscape had a clear shape: GPT-4 at the top, a large gap to everything else, and open-weights models unable to match it on anything rigorous. By early 2025, this had collapsed. Multiple models from multiple organizations match or exceed GPT-4 on standard benchmarks, open-weight models run locally on consumer hardware, and the "more compute = more capability" thesis — long used to justify proprietary advantages — no longer holds cleanly.

What changed was not a single breakthrough but a cluster of innovations, each addressing a different bottleneck. Understanding 2025 models means understanding which specific problem each architecture solved.

---

## 2. The Core Insight

Every 2025 frontier model solves one of three problems:

**Problem 1: KV cache costs too much at long context.** Standard multi-head attention caches K and V tensors for every head at every layer. At 128K tokens, this dominates GPU memory and limits batch size. Solutions: Multi-head Latent Attention (DeepSeek-V3), RoPE-free attention layers (Llama 4 iRoPE), Flash Attention variants.

**Problem 2: Supervised reasoning traces are expensive to collect.** Writing out worked solutions to hard math and code problems at scale requires either human annotation (slow) or distillation from closed models (licensing restrictions). Solution: pure RL training with verifiable rewards, where the model discovers reasoning strategies rather than imitating them (DeepSeek-R1, o3).

**Problem 3: Multimodal fusion is shallow when added post-hoc.** Bolting a vision encoder onto a pretrained language model via a linear projection means image and text representations are never jointly trained from the start. Solution: native early-fusion architectures where image and text tokens pass through the same transformer from layer 1 (Llama 4, Gemini 2.5, GPT-4o).

---

## 3. The Models

### DeepSeek-V3

**The KV cache problem in numbers:**

```
Standard MHA KV cache: 2 × n_heads × d_head × seq_len per layer
At 128K tokens, 128 heads, d_head=128: 2 × 128 × 128 × 128K = 4GB per layer
Across 60 layers: 240GB for KV cache alone
```

**Multi-head Latent Attention (MLA)** compresses K and V into a shared low-rank latent vector:

```
MLA forward:
  c_KV = W_DKV × x            # compress to low-rank latent
  K = W_UK × c_KV             # up-project to K space
  V = W_UV × c_KV             # up-project to V space
```

Only `c_KV` is cached, not full K and V. KV cache shrinks by ~13.5× at equivalent quality.

**Specs:** 671B total parameters, 37B active per token (MoE), 128 experts per layer, 128K context, trained on 14.8T tokens. Training cost: ~$5.5M on H800 GPUs — roughly 10–50× cheaper than comparable frontier models.

**Multi-Token Prediction (MTP):** Train auxiliary heads to predict tokens `t+2`, `t+3`, etc. simultaneously:

```
Main head:  P(x_{t+1} | x_{1:t})
Aux head 1: P(x_{t+2} | x_{1:t})
Aux head 2: P(x_{t+3} | x_{1:t})
```

At inference, auxiliary heads serve as speculative draft tokens — multiple tokens accepted per forward pass when auxiliary predictions are correct, implementing speculative decoding without a separate draft model.

---

### DeepSeek-R1

> For full GRPO derivation, Best-of-N, ORMs vs PRMs, and test-time compute scaling, see [large-reasoning-models.md](large-reasoning-models.md). This section focuses on R1's specific training pipeline and how it compares to o1/o3.

**The supervised trace problem:** Human-annotated chain-of-thought data for hard math and code costs roughly $1–10 per step. Getting millions of long reasoning traces this way is prohibitive. Distillation from OpenAI requires licensing. R1 is built on the insight that **reasoning can emerge from pure reinforcement learning with verifiable rewards**.

**Training recipe:**

**Phase 1 — Cold start SFT:** Fine-tune V3 on a small set of long chain-of-thought examples to establish a baseline reasoning format. Without this, pure RL produces incoherent traces.

**Phase 2 — GRPO (Group Relative Policy Optimization):**

```
For each prompt p:
1. Sample G rollouts from current policy: {o_1, ..., o_G}
2. Score each rollout with reward function r_i
3. Compute group baseline: r̄ = mean(r_1, ..., r_G)
4. Advantage for rollout i: A_i = (r_i - r̄) / std(r_1, ..., r_G)
5. Update policy using clipped PPO objective on A_i
```

No separate value (critic) network — the group serves as its own baseline, halving memory requirements vs PPO. Reward function: correct final answer = +1 for math; test suite pass rate for code.

**"Aha moment" behaviors** — spontaneously emerged from RL, not supervised:
- Backtracking: "Wait, let me reconsider..."
- Self-verification: checking calculations independently
- Extended reasoning chains: model learns longer traces correlate with correctness

**Phase 3 — Rejection sampling + SFT:** Collect high-quality reasoning traces by sampling many completions, keeping only correct ones. Fine-tune for stable reasoning.

**Phase 4 — Final RLHF:** Standard helpfulness alignment.

**R1 benchmarks vs o1:**
| Benchmark | o1 | DeepSeek-R1 |
|-----------|-----|-------------|
| AIME 2024 | 79.2% | 79.8% |
| MATH-500 | 96.4% | 97.3% |
| Codeforces (Elo) | 1891 | 2029 |
| MMLU | 91.8% | 90.8% |

R1 matches or exceeds o1 on math/code. Both models are fully open-weight.

**DeepSeek-R1-Distill Models:** R1-level reasoning distilled into smaller dense models via SFT on R1's reasoning traces. R1-Distill-Qwen-7B outperforms GPT-4o on AIME. Reasoning capability transfers via distillation — you don't need GRPO to get it.

---

### Meta Llama 4

**The shallow fusion problem:** LLaVA-style late fusion — frozen image encoder → linear projection → text LLM — is efficient but limits cross-modal interaction to whatever the projection can express. The LLM sees image features only as additional context tokens, without joint training.

**Native multimodality via early fusion:** Llama 4 processes image patches and text tokens in the same unified token stream from layer 1:

```
Images → patch embeddings (ViT-style)
Text  → token embeddings
Both → concatenated → full transformer stack from layer 1
```

This enables deeper cross-modal interaction but requires training from scratch on multimodal data.

**iRoPE (infinite RoPE):** Every 4th attention layer uses no positional encoding — pure attention without position bias. Remaining layers use standard RoPE. The hypothesis: some layers should be position-aware (local syntax) and some position-agnostic (global semantic matching). This enables arbitrarily long context without RoPE scaling hacks.

**Variants:**
- Scout: 17B active / 109B total — 10M context
- Maverick: 17B active / 400B total — 1M context
- Behemoth: ~2T total (unreleased)

All variants use MoE. Scout has 16 experts with top-1 routing, interleaved with dense layers at 1:3 ratio. 17B active params competitive with Llama 3 70B — ~7× parameter efficiency gain at similar quality.

---

### Google Gemini 2.0 / 2.5

**Gemini 2.0 Flash:** Natively multimodal (text, image, audio, video input), native tool use, real-time streaming for live audio/video, optional thinking tokens. The cost-efficient multimodal workhorse.

**Gemini 2.5 Pro (March 2025):**

1M token context window — the largest in production at launch. Enables loading an entire codebase, a full novel, or years of conversation history into context without retrieval. Extended "thinking" mode — internal CoT before response.

**Benchmark position (Q1 2025):**
| Task | Gemini 2.5 Pro |
|------|----------------|
| MMLU | 91.1% |
| HumanEval | 95.8% |
| SWE-Bench Verified | 63.8% |
| MATH (thinking) | 97.7% |

Architecture confirmed: natively multimodal unified encoder-decoder, MoE in at least some variants, efficient attention for long context.

---

### Anthropic Claude 3.7 Sonnet

**The always-on reasoning cost problem:** o1-style extended reasoning adds cost and latency to every query — including simple ones that don't need it. Claude 3.7 introduced **configurable extended thinking** — the developer controls when to pay the compute cost:

```python
import anthropic

client = anthropic.Anthropic()
response = client.messages.create(
    model="claude-3-7-sonnet-20250219",
    max_tokens=16000,
    thinking={
        "type": "enabled",
        "budget_tokens": 10000  # max thinking tokens
    },
    messages=[{"role": "user", "content": "Prove that sqrt(2) is irrational."}]
)

for block in response.content:
    if block.type == "thinking":
        print(f"Thinking: {block.thinking[:200]}...")
    else:
        print(f"Response: {block.text}")
```

`budget_tokens` controls maximum thinking compute. Higher budget = better on hard tasks; no benefit for simple factual queries.

**Benchmark position:**
| Benchmark | Claude 3.7 (thinking) | o1 |
|-----------|----------------------|-----|
| MATH | 96.2% | 96.4% |
| HumanEval | 93.1% | 92.4% |
| SWE-Bench Verified | 70.3% | 48.9% |
| MMLU | 91.1% | 91.8% |

**SWE-Bench 70.3%** is the headline number. SWE-Bench presents real GitHub issues; the model must produce a code patch that resolves the issue, validated by running the repository's test suite. 70.3% = 351 of 500 real software engineering problems correctly resolved. This drove adoption for agentic coding tools (Claude Code, Cursor, etc.).

Claude 3.7 in agentic settings: 200K context, strong tool call reliability, well-calibrated refusals.

---

### OpenAI GPT-4o & o3 / o4-mini

**GPT-4o:** Unified text, vision, and audio in a single model. Processes audio directly as audio tokens — enables emotional tone understanding and natural conversation pacing. Real-time audio I/O with ~320ms round-trip. Native multimodal.

**o3 (January 2025):** The flagship reasoning model, successor to o1.

ARC-AGI benchmark — a visual analogy reasoning benchmark designed to resist LLM pattern matching: GPT-4o scores ~5%; o3 scored **87.5%**. USAMO (competition math): o3 solved 25.2% vs o1's 13.4%.

Substantially more expensive than o1: o3-high is ~25× o1 cost. Reasoning depth is user-configurable at three levels.

**o4-mini (April 2025):** Smaller reasoning model that achieves o3-level performance on math/code at much lower cost. Reasoning capability transfers via distillation + RLVR to smaller models.

| Benchmark | o4-mini | o3 |
|-----------|---------|-----|
| AIME | 93.4% | 96.7% |
| Cost | ~60-70% cheaper | — |

---

### Alibaba Qwen3

**Overview (April 2025):** Flagship: **Qwen3-235B-A22B** (235B total, 22B active MoE). Thinking mode controlled by `/think` or `/no_think` system prompt tokens.

**Model family:**
| Model | Total Params | Active Params | Context |
|-------|-------------|---------------|---------|
| Qwen3-0.6B | 0.6B | 0.6B (dense) | 32K |
| Qwen3-4B | 4B | 4B (dense) | 32K |
| Qwen3-30B-A3B | 30B | 3B | 128K |
| Qwen3-235B-A22B | 235B | 22B | 128K |

**Qwen3-30B-A3B is the efficiency highlight** — 3B active parameters outperforming Qwen2.5-72B on several benchmarks. ~24× parameter efficiency ratio.

**Benchmark position (235B-A22B):**
| Benchmark | Qwen3-235B-A22B | DeepSeek-R1 | Claude 3.7 |
|-----------|----------------|-------------|------------|
| AIME 2024 | 85.7% | 79.8% | 96.2% (thinking) |
| LiveCodeBench | 70.7% | 65.9% | — |
| MMLU-Pro | 79.3% | 75.9% | 78.0% |

119-language support, strong on CJK. Fully open weights with commercial license.

---

### Mistral Large 3 & Codestral

**Mistral Large 3 (March 2025):** 128B dense model (not MoE). Why dense at 128B? MoE adds memory pressure (all experts must fit in GPU memory even when inactive) and communication overhead. For enterprise customers fine-tuning on dedicated hardware, dense models fine-tune more predictably. 128K context, strong multilingual (13 languages).

**Codestral 25.01:** 22B parameters, code-specialized. 256K context for full codebase awareness. Fill-in-the-middle (FIM) support — complete code given both prefix and suffix:

```python
prefix = "def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n"
suffix = "\n    return -1"
middle = codestral.complete(prefix=prefix, suffix=suffix)
```

FIM is critical for IDE integration — cursor-position completion requires knowing both what comes before and after the insertion point.

---

## 4. Model Selection

### By Use Case

| Use Case | Top Choice | Runner-up | Rationale |
|----------|-----------|-----------|-----------|
| **Hard math / reasoning** | Claude 3.7 + thinking | Gemini 2.5 Pro | SWE-Bench, MATH scores |
| **Code generation (agentic)** | Claude 3.7 | o4-mini | SWE-Bench 70% vs 48% |
| **Cost-efficient reasoning** | DeepSeek-R1 (open) | o4-mini | Open = free to self-host |
| **Long document analysis** | Gemini 2.5 Pro | Claude 3.7 | 1M context window |
| **Multimodal (image/audio)** | GPT-4o | Gemini 2.0 Flash | Native audio, mature API |
| **Edge / on-device** | Qwen3-4B | Llama 4 Scout | Small dense / efficient MoE |
| **Enterprise fine-tuning** | Mistral Large 3 | Llama 4 Scout | Dense = stable fine-tuning |
| **API cost sensitivity** | Gemini 2.0 Flash | o4-mini | Cheap + capable |
| **Open weights + reasoning** | DeepSeek-R1 | Qwen3-235B | Full open, self-hostable |
| **Real-time voice** | GPT-4o (audio) | Gemini 2.0 Flash Live | Native audio pipeline |

### Quality vs. Cost Tiers

```
Highest quality:  Claude 3.7 thinking / o3 / Gemini 2.5 Pro
Mid-tier:         GPT-4o / Claude 3.7 no-thinking / Qwen3-235B
Cost-efficient:   o4-mini / Gemini 2.0 Flash / DeepSeek-R1 (self-hosted)
Edge/local:       Qwen3-4B / Llama 4 Scout / R1-Distill-7B
```

---

## 5. What Breaks

**Benchmark saturation invalidates headline comparisons.** MMLU, HumanEval, and MATH are now near-saturated — most frontier models cluster at 90%+. SWE-Bench and ARC-AGI remain useful because they test actual task completion.

**MoE serving requires all experts in memory.** Qwen3-235B's 22B active params sounds cheap to serve, but all 235B parameters must be loaded. 4×A100 80GB is the minimum. "Active params" does not mean "memory used."

**Thinking mode does not help all tasks.** Extended reasoning benefits math, code, and multi-step logic. For factual retrieval and short-form generation, it adds latency and cost with no improvement. Setting `budget_tokens` too high on simple queries is a common mistake.

**Open-weight fine-tuning degrades MoE models unpredictably.** Dense models (Mistral Large 3, Llama 3 70B) fine-tune predictably. MoE models can have expert collapse or routing instability during fine-tuning, especially with small datasets or aggressive learning rates.

**Long context capability ≠ long context performance.** A 1M token context window does not mean the model accurately uses information at position 500K. Lost-in-the-middle degradation (U-shaped retrieval quality by position) is present in all current models.

**The "pure RL → emergent reasoning" narrative oversimplifies.** R1's cold-start SFT phase is load-bearing. The clean story of spontaneous emergence from RL training obscures a hybrid pipeline where the supervised phase shapes what RL discovers.

---

## Key Interview Points

- MLA reduces KV cache by ~13.5× vs MHA by caching a shared low-rank latent `c_KV` instead of full K and V tensors per head.
- GRPO eliminates the value network by using group-relative rewards: sample G rollouts per prompt, normalize advantages within the group. This halves memory vs PPO.
- Llama 4 early fusion = image patches and text tokens through the same transformer from layer 1. Requires training from scratch. Enables richer cross-modal representations than LLaVA-style projection.
- iRoPE: every 4th layer uses no positional encoding (position-agnostic semantic matching); remaining layers use RoPE. Enables 10M context without aggressive scaling hacks.
- Claude 3.7 `budget_tokens`: user-controlled compute budget for thinking. Higher budget = better on hard reasoning; no benefit for factual queries.
- SWE-Bench Verified: real GitHub issues, 500 problems, patch validated by running the repository's existing test suite. Claude 3.7 at 70.3%, o1 at 48.9%.
- o3 on ARC-AGI: 87.5% vs GPT-4o's ~5%. ARC-AGI tests visual analogy reasoning designed to resist LLM pattern matching.
- MTP (Multi-Token Prediction): trains auxiliary heads to predict t+2, t+3, etc. At inference, these serve as speculative draft tokens — multiple tokens accepted per forward pass without a separate draft model.

---

## Canonical Interview Q&As

**Q1: Derive how DeepSeek's Multi-Head Latent Attention achieves its KV cache compression. What are the exact memory savings and the mathematical cost?**

Standard MHA with H heads and d_head dimensions stores K and V tensors per layer:

```
KV memory per token = 2 × H × d_head bytes per layer
                    = 2 × 128 × 128 × 2 bytes  = 65,536 bytes/token (fp16)
```

MLA introduces a shared latent compression: instead of caching K and V separately, it caches a single compressed latent vector `c_KV`:

```
c_KV = W_DKV · h_t              (compress: d_model → d_c, d_c ≪ H·d_head)
K_t  = W_UK  · c_KV            (decompress at attention time)
V_t  = W_UV  · c_KV

Cache per token = d_c bytes  (single vector, not H pairs)
d_c = 512 in DeepSeek-V3   vs   H·d_head = 128×128 = 16,384
Compression ratio ≈ 16,384 / 512 = 32× per layer
```

There is also a decoupled RoPE head `k_R` (RoPE must be applied before caching, otherwise position IDs cannot be injected at serving time). This adds a small `d_Rh` component to the cache:

```
Cache per token per layer = d_c + d_Rh = 512 + 64 = 576 dims
vs. standard MHA = H × 2 × d_head = 128 × 2 × 128 = 32,768 dims
Net compression ≈ 32,768 / 576 ≈ 56.8× 
```

DeepSeek-V3 reports ~13.5× compared to standard MHA at their specific architecture dimensions. The key cost is the `W_UK / W_UV` decompression matmul at every attention step, which adds FLOPs relative to standard MHA but is acceptable because KV bandwidth (not FLOPs) is the inference bottleneck.

---

**Q2: GRPO eliminates the value network in RL post-training. Derive why this works and what it costs.**

Standard PPO requires four model copies in GPU memory: policy `π_θ`, reference `π_ref`, reward model `r_φ`, and value baseline `V_ψ`. The value network estimates `V(s)` to compute advantage `A = R - V(s)`, reducing policy gradient variance.

GRPO replaces the learned value baseline with a group-relative estimate:

```
For prompt x, sample G completions {y_1, ..., y_G}
Compute rewards {r_1, ..., r_G}

Group-relative advantage:
  Â_i = (r_i - mean({r_j})) / std({r_j})

GRPO objective:
  L_GRPO = E_i [ min(ratio_i · Â_i, clip(ratio_i, 1-ε, 1+ε) · Â_i) - β·KL(π_θ||π_ref) ]
  where ratio_i = π_θ(y_i|x) / π_old(y_i|x)
```

Why this is valid: the value function `V(s)` is a control variate that reduces variance without introducing bias. The group mean `mean(r_j)` is also a valid control variate — it is unbiased as long as the G samples are drawn i.i.d. from the same prompt. Variance reduction equals that of the sample mean of G rewards, which is `Var(R)/G`.

Cost: you need G rollouts per training step (G = 8–16 in R1 training), which multiplies generation compute. But you save one entire model copy in memory and eliminate value network training instability. For reasoning tasks with verifiable rewards (math answers, code execution), G rollouts are cheap relative to the stability gain.

---

**Q3: What does Llama 4's "early fusion" mean architecturally, and why does it require training from scratch rather than adapting an existing LLM?**

Late fusion (LLaVA-style): a frozen vision encoder (ViT) maps images to embedding vectors, an MLP projection aligns them to text embedding space, and the LLM processes the result as if they were token embeddings. Modalities are separate until the LLM's input layer.

```
Image → ViT → [v_1, ..., v_N] → MLP_proj → [e_1, ..., e_N]
Text  → tokenizer              → [e_{N+1}, ..., e_M]
LLM input: [e_1, ..., e_N, e_{N+1}, ..., e_M]
```

Early fusion: image patches are tokenized into the same vocabulary as text (or via a separate but jointly trained encoder), and the transformer attends over interleaved image and text tokens from layer 1:

```
All layers: attention(Q, K, V) where K, V come from both image and text positions
Cross-modal interaction at every layer, not just at input
```

Why it requires training from scratch: a pre-trained LLM has developed weight matrices (especially in attention and FFN) that are specialized for text token distributions. Image token distributions are fundamentally different — different statistics, different semantic structure. Injecting image tokens into a frozen text-trained model produces systematic representation mismatch that cannot be corrected by fine-tuning the projection layer alone. The attention patterns, positional encodings, and residual stream norms must all co-adapt to the joint distribution from the start.

The key evidence: LLaVA-style models with frozen LLMs show weaker spatial reasoning and object relationship understanding than early-fusion models, because the latter can develop cross-modal attention patterns at every layer rather than a single point of injection.

---

**Q4: Claude 3.7's extended thinking uses `budget_tokens` for user-controlled compute. Design a production system that uses this parameter efficiently across a mixed query workload.**

The key insight: thinking helps on complex multi-step tasks and hurts cost/latency on simple ones. The system must classify queries before setting `budget_tokens`.

```python
ROUTING_RULES = {
    "factual_retrieval":  {"budget_tokens": 0,    "reason": "no reasoning needed"},
    "summarization":      {"budget_tokens": 0,    "reason": "linear task"},
    "code_completion":    {"budget_tokens": 2000, "reason": "moderate complexity"},
    "math_proof":         {"budget_tokens": 8000, "reason": "multi-step deductive"},
    "agentic_planning":   {"budget_tokens": 16000,"reason": "tree search over actions"},
    "adversarial_eval":   {"budget_tokens": 32000,"reason": "maximize correctness"},
}

def classify_query(query: str, context: dict) -> str:
    # Fast classifier (fine-tuned Haiku or heuristic) estimates task type
    # Features: question type, entity count, presence of math symbols, code fence
    ...

def get_budget(query: str, context: dict, cost_target: float) -> int:
    task_type = classify_query(query, context)
    base_budget = ROUTING_RULES[task_type]["budget_tokens"]
    # Scale down under cost pressure; minimum 0 (disable thinking)
    return min(base_budget, int(cost_target * MAX_BUDGET))
```

Additional production considerations:
- Thinking tokens are billed but not returned to the user by default. Monitor `usage.cache_read_input_tokens` to understand thinking overhead.
- For batch workloads, use `budget_tokens=0` and post-process with a separate verification pass only on low-confidence outputs.
- Cache hit rate degrades with extended thinking because the thinking prefix is unique per query. Design prompts to maximize the shared prefix for cache warming.
- SWE-Bench result (70.3% with thinking vs. ~40% without) is the reference calibration: thinking roughly doubles code task accuracy at 5–10× cost.

---

**Q5: MoE serving requires all expert weights in memory. Design a serving system for Qwen3-235B-A22B that minimizes cost while meeting <2 second TTFT at p95.**

Problem: Qwen3-235B-A22B has 235B total parameters, ~470GB in fp16. At 80GB A100s, that's a minimum of 6 GPUs just for weights. But active parameters per token are 22B, so the compute utilization is ~22/235 ≈ 9%.

Architecture for efficient serving:

```
Tier 1: High-Memory Low-GPU-Count machines (H100 NVL 94GB)
  - 4×H100 NVL = 376GB. Not enough for fp16. Use INT4 = 235B × 0.5 bytes = 117GB → fits on 2×H100.
  - Expert routing via learned router in INT8 for precision.

Tier 2: Expert Parallelism
  - Router runs on CPU / small shared GPU.
  - Each GPU holds a subset of experts. Token routing dispatches to the right GPU.
  - All-to-all communication at each MoE layer. Use NVLink (600 GB/s) not PCIe (16 GB/s).

Serving strategy for <2s TTFT p95:
  1. Continuous batching: batch size 8-16 for MoE (larger than dense, because routing amortizes).
  2. Expert capacity buffer: top-2 routing with overflow buffer to prevent token dropping.
  3. KV cache offloading: 22B active means KV cache is manageable; HBM-resident for hot context.
  4. Quantization: GPTQ INT4 for experts (high-volume, low-precision-need) + INT8 for attention.
```

Key numbers: Qwen3-235B-A22B at INT4 ≈ 117GB → 2×H100 NVL minimum. At batch=16, TTFT dominated by prefill compute at 22B active params ≈ fast enough for <2s at 4K input length. For 32K input, need expert parallelism across 4 GPUs to meet SLA.

## Flashcards

**Backtracking?** #flashcard
"Wait, let me reconsider..."

**Self-verification?** #flashcard
checking calculations independently

**Extended reasoning chains?** #flashcard
model learns longer traces correlate with correctness

**Scout: 17B active / 109B total?** #flashcard
10M context

**Maverick: 17B active / 400B total?** #flashcard
1M context

**Behemoth?** #flashcard
~2T total (unreleased)

**MLA reduces KV cache by ~13.5× vs MHA by caching a shared low-rank latent c_KV instead of full K and V tensors per head.?** #flashcard
MLA reduces KV cache by ~13.5× vs MHA by caching a shared low-rank latent c_KV instead of full K and V tensors per head.

**GRPO eliminates the value network by using group-relative rewards?** #flashcard
sample G rollouts per prompt, normalize advantages within the group. This halves memory vs PPO.

**Llama 4 early fusion = image patches and text tokens through the same transformer from layer 1. Requires training from scratch. Enables richer cross-modal representations than LLaVA-style projection.?** #flashcard
Llama 4 early fusion = image patches and text tokens through the same transformer from layer 1. Requires training from scratch. Enables richer cross-modal representations than LLaVA-style projection.

**iRoPE?** #flashcard
every 4th layer uses no positional encoding (position-agnostic semantic matching); remaining layers use RoPE. Enables 10M context without aggressive scaling hacks.

**Claude 3.7 budget_tokens?** #flashcard
user-controlled compute budget for thinking. Higher budget = better on hard reasoning; no benefit for factual queries.

**SWE-Bench Verified?** #flashcard
real GitHub issues, 500 problems, patch validated by running the repository's existing test suite. Claude 3.7 at 70.3%, o1 at 48.9%.

**o3 on ARC-AGI?** #flashcard
87.5% vs GPT-4o's ~5%. ARC-AGI tests visual analogy reasoning designed to resist LLM pattern matching.

**MTP (Multi-Token Prediction): trains auxiliary heads to predict t+2, t+3, etc. At inference, these serve as speculative draft tokens?** #flashcard
multiple tokens accepted per forward pass without a separate draft model.

**Thinking tokens are billed but not returned to the user by default. Monitor usage.cache_read_input_tokens to understand thinking overhead.?** #flashcard
Thinking tokens are billed but not returned to the user by default. Monitor usage.cache_read_input_tokens to understand thinking overhead.

**For batch workloads, use budget_tokens=0 and post-process with a separate verification pass only on low-confidence outputs.?** #flashcard
For batch workloads, use budget_tokens=0 and post-process with a separate verification pass only on low-confidence outputs.

**Cache hit rate degrades with extended thinking because the thinking prefix is unique per query. Design prompts to maximize the shared prefix for cache warming.?** #flashcard
Cache hit rate degrades with extended thinking because the thinking prefix is unique per query. Design prompts to maximize the shared prefix for cache warming.

**SWE-Bench result (70.3% with thinking vs. ~40% without) is the reference calibration?** #flashcard
thinking roughly doubles code task accuracy at 5–10× cost.

**4×H100 NVL = 376GB. Not enough for fp16. Use INT4 = 235B × 0.5 bytes = 117GB → fits on 2×H100.?** #flashcard
4×H100 NVL = 376GB. Not enough for fp16. Use INT4 = 235B × 0.5 bytes = 117GB → fits on 2×H100.

**Expert routing via learned router in INT8 for precision.?** #flashcard
Expert routing via learned router in INT8 for precision.

**Router runs on CPU / small shared GPU.?** #flashcard
Router runs on CPU / small shared GPU.

**Each GPU holds a subset of experts. Token routing dispatches to the right GPU.?** #flashcard
Each GPU holds a subset of experts. Token routing dispatches to the right GPU.

**All-to-all communication at each MoE layer. Use NVLink (600 GB/s) not PCIe (16 GB/s).?** #flashcard
All-to-all communication at each MoE layer. Use NVLink (600 GB/s) not PCIe (16 GB/s).
