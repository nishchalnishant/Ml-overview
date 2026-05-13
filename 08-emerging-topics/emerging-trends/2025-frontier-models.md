# 2025 Frontier Models — Field Guide

> Interview-grade reference for the model releases that reshuffled the rankings in 2025. Covers architecture deltas, benchmark significance, and "which model for which job" decision logic.

---

## Table of Contents

1. [DeepSeek-V3 & R1](#1-deepseek-v3--r1)
2. [Meta Llama 4](#2-meta-llama-4)
3. [Google Gemini 2.0 / 2.5](#3-google-gemini-20--25)
4. [Anthropic Claude 3.7 Sonnet](#4-anthropic-claude-37-sonnet)
5. [OpenAI GPT-4o & o3 / o4-mini](#5-openai-gpt-4o--o3--o4-mini)
6. [Alibaba Qwen3](#6-alibaba-qwen3)
7. [Mistral Large 3 & Codestral](#7-mistral-large-3--codestral)
8. [Model Selection Decision Guide](#8-model-selection-decision-guide)
9. [Common Interview Questions](#9-common-interview-questions)

---

## 1. DeepSeek-V3 & R1

### Why This Matters

DeepSeek (Hangzhou AI lab backed by High-Flyer quant fund) released two models in late 2024 / early 2025 that fundamentally changed the open-weights landscape: **DeepSeek-V3** (a frontier-grade dense-MoE base model) and **DeepSeek-R1** (a reasoning model matching OpenAI o1 on math/code benchmarks). Both are fully open-weight.

The geopolitical significance: these models were trained on H800 GPUs (export-controlled Hopper, not full H100) at a fraction of GPT-4 training cost, raising serious questions about the "compute = capability" thesis.

---

### DeepSeek-V3 Architecture

**Key specs:**
- 671B total parameters, 37B active per token (MoE)
- 61 transformer layers
- 128 experts per MoE layer, top-2 routing (but see MLA below)
- 128K token context window
- Trained on 14.8T tokens

**Multi-head Latent Attention (MLA):**
The flagship architectural innovation. Standard KV cache stores K and V tensors per head per layer — expensive at long context. MLA compresses the KV cache into a **low-rank latent vector**:

```
Standard KV: cache size = 2 × n_heads × d_head × seq_len
MLA:         cache size = d_compressed × seq_len  (d_compressed << 2 × n_heads × d_head)

MLA forward:
  c_KV = W_DKV × x            # compress to low-rank latent
  K = W_UK × c_KV             # up-project to K space
  V = W_UV × c_KV             # up-project to V space
```

Only `c_KV` (the latent) is cached, not full K and V. This reduces KV cache size by ~13.5× vs MHA at the same quality level — a major throughput win at long context.

**Multi-Token Prediction (MTP):**
Instead of predicting only the next token, DeepSeek-V3 is trained to predict the next **N tokens** simultaneously using auxiliary prediction heads:

```
Main head:  P(x_{t+1} | x_{1:t})
Aux head 1: P(x_{t+2} | x_{1:t})
Aux head 2: P(x_{t+3} | x_{1:t})
```

At inference, MTP heads serve as speculative draft tokens — the main model can accept multiple tokens per forward pass if the auxiliary predictions are correct (effectively a built-in speculative decoding mechanism).

**Training efficiency:**
DeepSeek-V3 was trained for ~2.79M H800 GPU-hours at a reported cost of ~$5.5M — roughly 10–50× cheaper than comparable frontier models. Key techniques: FP8 mixed precision, DualPipe pipeline schedule (overlaps forward/backward with communication), and a custom all-to-all communication kernel for expert routing.

---

### DeepSeek-R1 Training Recipe

R1 is a reasoning model built on top of DeepSeek-V3 using **pure RL** (no supervised reasoning trace examples):

**Phase 1 — Cold start SFT:**
Fine-tune V3 on a small set of long chain-of-thought examples to establish a baseline reasoning format. Without this, pure RL produces incoherent reasoning traces.

**Phase 2 — GRPO (Group Relative Policy Optimization):**

```
For each prompt p:
1. Sample G rollouts from current policy: {o_1, ..., o_G}
2. Score each rollout with reward function r_i
3. Compute group baseline: r̄ = mean(r_1, ..., r_G)
4. Advantage for rollout i: A_i = (r_i - r̄) / std(r_1, ..., r_G)
5. Update policy using clipped PPO objective on A_i
```

Key insight: no separate value (critic) network. The group itself serves as the baseline — halving memory requirements. The reward function for math: verifiable correct final answer = +1. For code: test suite pass rate.

**"Aha moment" behaviors** — spontaneous emergence (not supervised):
- Backtracking: "Wait, let me reconsider..."
- Self-verification: checking calculations independently
- Extended reasoning chains: model learns to write longer traces when they correlate with correctness

**Phase 3 — Rejection sampling + SFT:**
After GRPO, collect high-quality reasoning traces by sampling many completions and keeping only correct ones. Fine-tune on these to make reasoning more stable.

**Phase 4 — Final RLHF:**
Standard helpfulness alignment on top of the reasoning model.

**R1 benchmarks vs o1:**
| Benchmark | o1 | DeepSeek-R1 |
|-----------|-----|-------------|
| AIME 2024 | 79.2% | 79.8% |
| MATH-500 | 96.4% | 97.3% |
| Codeforces (Elo) | 1891 | 2029 |
| MMLU | 91.8% | 90.8% |

R1 matches or exceeds o1 on math/code. Slightly behind on knowledge tasks (MMLU).

---

### DeepSeek-R1-Distill Models

R1-level reasoning distilled into smaller dense models via supervised fine-tuning on R1's reasoning traces:
- R1-Distill-Qwen-7B: outperforms GPT-4o on AIME, Math-500
- R1-Distill-Llama-70B: best open reasoning model at 70B scale

The distillation insight: you don't need GRPO to get reasoning capabilities — training a small model on long-CoT outputs from a reasoning model transfers the capability efficiently.

---

## 2. Meta Llama 4

### Overview

Meta released Llama 4 in April 2025 with a significant architectural departure: **native multimodality** and **mixture-of-experts at all scales**. Three variants: Scout (17B active / 109B total), Maverick (17B active / 400B total), and the unreleased Behemoth (~2T total).

---

### Architecture Changes vs Llama 3

**From dense to MoE:** Every Llama 4 variant uses MoE. Llama 3 was dense; Llama 4 Scout has 16 experts with top-1 routing (interleaved with dense layers at 1:3 ratio).

**Native multimodality via early fusion:** Unlike LLaVA-style late fusion (image encoder → projection → LLM), Llama 4 processes image patches and text tokens in the **same unified token stream from layer 1**:

```
Images → patch embeddings (ViT-style)
Text  → token embeddings
Both → concatenated → full transformer stack from layer 1
```

This allows deeper cross-modal interaction but requires training from scratch on multimodal data (can't bolt a vision encoder onto a pretrained language model).

**iRoPE (infinite RoPE):** Every 4th attention layer uses **no positional encoding** — pure attention without position bias. Remaining layers use standard RoPE. The hypothesis: some layers should be position-aware (for local syntax) and some should be position-agnostic (for global semantic matching). This enables arbitrarily long context without RoPE scaling hacks.

**Context:** Scout — 10M tokens; Maverick — 1M tokens.

---

### Benchmark Position

Maverick is competitive with GPT-4o and Claude Sonnet 3.7 on most benchmarks. Scout punches well above its active parameter count, competitive with Llama 3 70B at 17B active params.

**Scout efficiency:** 17B active params with 109B total = ~7× parameter efficiency gain vs dense 70B, serving at 70B-competitive quality.

---

## 3. Google Gemini 2.0 / 2.5

### Gemini 2.0 Flash (December 2024)

The production workhorse of the Gemini family. Key additions over 1.5 Flash:
- **Native tool use:** model can call tools (search, code execution, image generation) as a first-class capability, not an afterthought
- **Real-time streaming:** designed for live audio/video applications
- **Thinking tokens (optional):** can enable extended reasoning with a budget
- Multimodal: text, image, audio, video input; text + image output

**2.0 Flash is Gemini's answer to Claude Haiku** — cheap, fast, surprisingly capable, multimodal from the start.

---

### Gemini 2.5 Pro (March 2025)

The flagship reasoning model. Key specs:
- **1M token context window** — the largest in production at launch
- **"Thinking" mode** — extended internal CoT before response (similar to o1/R1)
- Best-in-class on coding benchmarks (SWE-Bench Verified: 63.8% at launch)
- Strong on long-document QA (RULER, SCROLLS)

**The 1M context significance:** Enables loading an entire codebase, a full novel, or years of conversation history into context. Combined with native multimodal, this creates genuinely new use cases:
- Full codebase analysis without retrieval
- Video understanding at film length
- Long-context scientific paper synthesis

**Benchmark position (as of Q1 2025):**
| Task | Gemini 2.5 Pro |
|------|----------------|
| MMLU | 91.1% |
| HumanEval | 95.8% |
| SWE-Bench Verified | 63.8% |
| MATH | 97.7% (thinking mode) |

---

### Gemini Architecture Notes

Gemini uses a **natively multimodal architecture** — all modalities are processed in a unified encoder-decoder framework, not a text LLM with bolted-on vision. Specific architecture details are not public, but Gemini 1.5 technical report confirms:
- Efficient attention for long context (linear or near-linear complexity)
- Mixture-of-Experts in at least some variants
- SentencePiece tokenizer with multimodal tokens interleaved

---

## 4. Anthropic Claude 3.7 Sonnet

### Extended Thinking

Claude 3.7 Sonnet (February 2025) introduced **extended thinking** — a configurable reasoning mode where the model generates an internal scratchpad before the final response.

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

# Response has two blocks:
# 1. ThinkingBlock (internal reasoning, not shown to user by default)
# 2. TextBlock (final response)
for block in response.content:
    if block.type == "thinking":
        print(f"Thinking: {block.thinking[:200]}...")
    else:
        print(f"Response: {block.text}")
```

**Hybrid mode:** Claude 3.7 can switch between fast (no thinking) and extended thinking within a session. Unlike o1 which always reasons, 3.7 gives developers control over when to pay the compute cost.

**Budget tokens:** The `budget_tokens` parameter controls how much thinking compute the model uses. Higher budget = better performance on hard tasks, but higher latency and cost. Empirically, math and code tasks benefit most; simple factual tasks don't benefit.

---

### Benchmark Position

| Benchmark | Claude 3.7 (thinking) | o1 |
|-----------|----------------------|-----|
| MATH | 96.2% | 96.4% |
| HumanEval | 93.1% | 92.4% |
| SWE-Bench Verified | 70.3% | 48.9% |
| MMLU | 91.1% | 91.8% |

The **SWE-Bench number is the headline** — 70.3% on real GitHub issue resolution is the highest published score, significantly above o1. This drove adoption for agentic coding workflows (Cursor, Claude Code, etc.).

---

### Claude 3.7 in Agentic Settings

The 200K context window and strong instruction following make 3.7 the current default for complex agentic tasks. Key properties that matter for agents:
- **Tool call reliability** — low rate of malformed tool invocations
- **Context utilization** — doesn't lose track of early instructions in long agentic loops
- **Safety calibration** — well-calibrated refusals that don't over-refuse in legitimate agentic contexts

---

## 5. OpenAI GPT-4o & o3 / o4-mini

### GPT-4o (May 2024 → ongoing updates)

GPT-4o ("omni") unified OpenAI's text, vision, and audio capabilities into a single model with native multimodal processing. Key properties:
- Processes text, images, and audio as native inputs (not separate pipelines)
- Real-time audio I/O with human-like latency (~320ms round-trip)
- Vision substantially improved over GPT-4V — fine-grained OCR, chart analysis, spatial reasoning
- Faster and cheaper than GPT-4 Turbo at higher quality

**Architecture note:** GPT-4o processes audio directly as audio tokens, not text transcripts — this enables emotional tone understanding and natural conversation pacing.

---

### o3 (January 2025)

OpenAI's flagship reasoning model, successor to o1. Not yet publicly available as a standalone product (released via API with usage restrictions); powers ChatGPT Pro mode.

**ARC-AGI benchmark (o3):**
The ARC-AGI benchmark (Abstraction and Reasoning Corpus) was designed by François Chollet to be resistant to LLM pattern matching — it tests novel visual analogy reasoning. o3 achieved **87.5% on the semi-private ARC-AGI eval** (vs ~5% for GPT-4o), widely reported as a significant step toward general problem solving.

**How o3 differs from o1:**
- More reasoning compute at inference (longer thinking traces)
- Better verified reasoning — less hallucination in multi-step proofs
- Substantially more expensive (o3-high usage is ~25× o1 cost)

**USAMO (math olympiad):** o3 solved 25.2% of USAMO problems vs o1's 13.4% — doubling performance on competition-level math.

---

### o4-mini (April 2025)

A smaller, faster, cheaper reasoning model that achieves o3-level performance on math/code with much lower latency and cost. The key insight from o-series: **reasoning capability transfers to smaller models** via distillation + RLVR.

o4-mini vs o3 on key benchmarks:
- AIME: o4-mini 93.4% vs o3 96.7%
- Codeforces: o4-mini competitive with o3
- Cost: ~60-70% cheaper than o3

**Production implication:** For math-heavy or code-generation workloads, o4-mini gives the best cost-efficiency. o3/o3-pro for genuinely novel problem solving.

---

## 6. Alibaba Qwen3

### Overview (April 2025)

Qwen3 is Alibaba's third-generation open-weight model family — the most powerful open-weights release as of Q2 2025. The flagship is **Qwen3-235B-A22B** (235B total, 22B active MoE).

---

### Architecture

**Thinking / non-thinking modes:** Like DeepSeek-R1 and Gemini 2.5, Qwen3 has explicit thinking mode control. In non-thinking mode it's a fast general assistant; in thinking mode it generates extended reasoning traces. The switch is controlled by a special token `/think` or `/no_think` in the system prompt.

**MoE at all scales:** The full model family:
| Model | Total Params | Active Params | Context |
|-------|-------------|---------------|---------|
| Qwen3-0.6B | 0.6B | 0.6B (dense) | 32K |
| Qwen3-4B | 4B | 4B (dense) | 32K |
| Qwen3-30B-A3B | 30B | 3B | 128K |
| Qwen3-235B-A22B | 235B | 22B | 128K |

**Qwen3-30B-A3B is the efficiency highlight** — 3B active parameters outperforming Qwen2.5-72B on several benchmarks. A ~24× parameter efficiency ratio.

---

### Benchmark Position (235B-A22B)

| Benchmark | Qwen3-235B-A22B | DeepSeek-R1 | Claude 3.7 |
|-----------|----------------|-------------|------------|
| AIME 2024 | 85.7% | 79.8% | 96.2% (thinking) |
| LiveCodeBench | 70.7% | 65.9% | — |
| MMLU-Pro | 79.3% | 75.9% | 78.0% |

Strong across-the-board performance, particularly on code benchmarks.

---

### Why Qwen3 Matters for the Ecosystem

1. **Multilingual:** 119-language support, strong on CJK (Chinese-Japanese-Korean) — addresses the English-centric bias of most frontier models
2. **Fully open weights + commercial license** — can be fine-tuned and deployed commercially without API costs
3. **Efficient serving:** 22B active params means a 235B model can be served on 4×A100 80GB with reasonable throughput

---

## 7. Mistral Large 3 & Codestral

### Mistral Large 3 (March 2025)

Mistral's flagship 128B dense model. Positioned as the premium open-weight alternative to proprietary APIs:
- 128B parameters (dense, not MoE)
- 128K context
- Strong multilingual (13 languages)
- Best-in-class for enterprise fine-tuning (large dense models fine-tune more predictably than MoE)

**Why dense at 128B?** MoE adds memory pressure (all experts must fit in GPU memory even when inactive) and communication overhead. For enterprise customers who need to fine-tune and run inference on dedicated hardware, a dense model is often operationally simpler.

---

### Codestral 25.01

Mistral's code-specialized model. Key properties:
- 22B parameters, optimized for code generation
- Fill-in-the-middle (FIM) support — can complete code given both prefix and suffix
- 256K context for full codebase awareness
- Benchmark: 78.4% on HumanEval

**FIM example:**
```python
# Prefix: the code before the cursor
prefix = "def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n"

# Suffix: the code after the cursor
suffix = "\n    return -1"

# Model fills the middle
middle = codestral.complete(prefix=prefix, suffix=suffix)
```

FIM is critical for IDE integration — Cursor, VS Code Copilot, and similar tools need to complete code at the cursor position, not just at the end of a file.

---

## 8. Model Selection Decision Guide

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

---

### Key Trade-off Axes

**Quality vs. cost:**
```
Highest quality:  Claude 3.7 thinking / o3 / Gemini 2.5 Pro
Mid-tier:         GPT-4o / Claude 3.7 no-thinking / Qwen3-235B
Cost-efficient:   o4-mini / Gemini 2.0 Flash / DeepSeek-R1 (self-hosted)
Edge/local:       Qwen3-4B / Llama 4 Scout / R1-Distill-7B
```

**Open vs. proprietary:**
- Open (self-hostable): DeepSeek-V3/R1, Llama 4, Qwen3, Mistral Large 3, Codestral
- Proprietary API only: GPT-4o, o3, Claude 3.7, Gemini 2.5

**Latency vs. reasoning depth:**
```
Fastest:          Gemini 2.0 Flash, GPT-4o-mini
Balanced:         Claude 3.7 (no-thinking), GPT-4o
Extended reasoning: Claude 3.7 + thinking, Gemini 2.5 + thinking, o3
```

---

## 9. Common Interview Questions

**Q: What is the key architectural innovation in DeepSeek-V3 that reduces serving cost?**

Multi-head Latent Attention (MLA). Standard attention caches K and V tensors per head — O(n_heads × d_head) memory per token. MLA projects K and V down to a shared low-rank latent vector `c_KV`, and only caches that. At inference, K and V are reconstructed from `c_KV` on the fly. The KV cache is ~13.5× smaller than equivalent MHA, enabling larger batch sizes and longer effective context at the same GPU memory budget.

---

**Q: DeepSeek-R1 uses GRPO instead of PPO. What's the practical difference?**

PPO requires a separate value network (critic) to estimate the expected reward from each state — this doubles memory and compute requirements during training. GRPO eliminates the critic by using group statistics: sample G rollouts per prompt, compute each rollout's reward, then use the group mean as the baseline and group std for normalization. The relative reward within the group is the advantage signal. Same convergence behavior as PPO for language model tasks, half the memory footprint.

---

**Q: Why did Llama 4 switch to early fusion for multimodality?**

Late fusion (LLaVA-style: frozen image encoder → linear projection → frozen LLM) is efficient but shallow — the LLM sees image features only as additional context tokens, without the image and text representations being jointly trained from the beginning. Early fusion processes image patches and text tokens through the same transformer from layer 1, allowing the model to learn richer cross-modal representations. The downside: must train from scratch (can't reuse a pretrained language backbone), requiring more compute and multimodal data.

---

**Q: What is the iRoPE positional encoding used in Llama 4?**

iRoPE applies standard RoPE to 3 out of every 4 attention layers, and no positional encoding to the remaining 1 in 4 layers. The layers without positional encoding perform purely content-based attention — useful for global semantic matching where position shouldn't matter (e.g., finding a relevant fact regardless of where in the document it appears). The layers with RoPE handle position-sensitive tasks (local syntax, relative ordering). This combination enables the 10M token context in Scout without requiring aggressive RoPE scaling, and the "infinite" in iRoPE refers to the position-agnostic layers which theoretically generalize to arbitrary lengths.

---

**Q: How do thinking/extended reasoning models decide how much to think?**

Different approaches:
- **o1/o3**: Thinking is always on; compute budget is set by the model based on perceived difficulty (not user-controllable at the prompt level)
- **Claude 3.7**: `budget_tokens` parameter — user sets the maximum number of thinking tokens; model uses up to that budget
- **Gemini 2.5**: Thinking mode enabled/disabled; thinking budget configurable
- **Qwen3**: `/think` and `/no_think` system prompt tokens toggle thinking mode
- **DeepSeek-R1**: Thinking is baked in as a `<think>...</think>` XML block in the response format; always present for the reasoning models

The key trade-off: thinking mode improves performance on hard reasoning tasks (math, code, logic) but adds latency and cost proportional to the thinking token count. For simple factual queries, thinking mode adds overhead with no benefit.

---

**Q: What does the SWE-Bench Verified benchmark actually measure?**

SWE-Bench (Software Engineering Benchmark) presents models with real GitHub issues from popular open-source Python repositories, including the full repository codebase as context. The model must produce a code patch that resolves the issue. The patch is then applied to the repository and the existing test suite is run — if the tests pass (and previously failing tests now pass), it counts as resolved. SWE-Bench Verified is a cleaned subset (500 problems) where human annotators confirmed the issues are solvable and the test suites reliably verify the fix. A 70% score means the model correctly resolved 350 of 500 real-world software engineering tasks without human guidance.

---

**Q: What is Multi-Token Prediction and why does it help inference speed?**

Standard LLMs are trained with a next-token prediction objective: predict token `t+1` given tokens `1..t`. MTP adds auxiliary prediction heads that simultaneously predict tokens `t+2`, `t+3`, etc. during training. At inference, the auxiliary heads serve as speculative draft tokens: after each forward pass, you get candidate next tokens from all heads. You then run a verification step to see how many of the speculative tokens match the model's actual distribution. Accepted tokens advance the sequence by multiple positions per forward pass — effectively implementing speculative decoding without needing a separate draft model.

---

*Last updated: May 2026 | Coverage: 2024–2025 frontier model releases | Focus: interview precision, architecture deltas, decision logic*
