---
module: References
topic: Research Papers
subtopic: Deep Learning Llm
status: unread
tags: [references, ml, research-papers-deep-learning]
---
# LLM Research Papers — Foundational Reading

> 25 papers every ML engineer should know. One-sentence contribution, key result, and interview relevance for each.

---

## How to Use This List

**Tier 1 (Must know):** Can explain the core idea, key equation, and why it mattered.
**Tier 2 (Should know):** Can summarize the contribution and benchmark results.
**Tier 3 (Know it exists):** Can name it and place it in the timeline.

---

## Foundation Architecture

### [Tier 1] Attention Is All You Need (Vaswani et al., 2017)
**Link:** https://arxiv.org/abs/1706.03762

**Contribution:** Introduced the Transformer — self-attention replaces recurrence entirely. Multi-head attention, positional encodings, encoder-decoder with cross-attention.

**Key result:** State-of-art on WMT English-German translation (28.4 BLEU) with 3.5× less training time than best RNN models.

**Core equation:**
```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

**Why it matters:** Every modern LLM (GPT, BERT, T5, LLaMA) is a transformer. This is the foundational paper.

**Interview hook:** "Why scale Q·K^T by √d_k?" — dot products grow in magnitude with dimension d_k, pushing softmax into saturation (near-zero gradients). √d_k rescaling keeps the variance of dot products ~1.

---

### [Tier 1] BERT (Devlin et al., 2018)
**Link:** https://arxiv.org/abs/1810.04805

**Contribution:** Bidirectional encoder pre-training via Masked Language Modeling (MLM) and Next Sentence Prediction (NSP). Demonstrates transfer learning from unsupervised text → fine-tune on downstream tasks.

**Key result:** 11 NLP tasks state-of-art simultaneously. GLUE benchmark: 80.5% (vs 69.1% prior SOTA).

**Interview hook:** Why is BERT encoder-only? Causal masking isn't needed for understanding tasks — bidirectional context improves representations for classification, NER, QA.

---

### [Tier 1] GPT-3 (Brown et al., 2020)
**Link:** https://arxiv.org/abs/2005.14165

**Contribution:** Demonstrated **in-context learning** — a 175B parameter model can perform few-shot tasks from examples in the prompt, with no gradient updates. Scaling alone unlocks new capabilities.

**Key result:** Few-shot SuperGLUE: 71.8% (vs 89.3% fine-tuned BERT — but GPT-3 never sees training examples for the task).

**Interview hook:** In-context learning isn't gradient descent on the weights — it's Bayesian inference over implicit task hypotheses given the prompt examples. The model's weights are fixed; the "learning" is in the forward pass.

---

## Scaling Laws

### [Tier 1] Scaling Laws for Neural Language Models (Kaplan et al., 2020)
**Link:** https://arxiv.org/abs/2001.08361

**Contribution:** Test loss follows power laws in model size N, data D, and compute C. For a fixed compute budget, scaling model size dominates — led to GPT-3.

**Key result:** `L(N) ~ N^{-0.076}`, `L(D) ~ D^{-0.095}`. Compute-optimal: maximize model size given C.

---

### [Tier 1] Training Compute-Optimal LLMs / Chinchilla (Hoffmann et al., 2022)
**Link:** https://arxiv.org/abs/2203.15556

**Contribution:** Showed Kaplan's compute model was flawed. True compute-optimal: scale parameters and tokens equally. The ~20 tokens/parameter rule.

**Key result:** Chinchilla 70B (1.4T tokens) outperforms Gopher 280B (300B tokens) at equal compute — proving GPT-3 was massively under-trained relative to its compute budget.

**Interview hook:** "Why do LLaMA models over-train beyond Chinchilla-optimal?" — inference efficiency. Smaller models serve cheaper; it's worth paying extra training compute to get a smaller model that serves at the same quality.

---

## Alignment & Fine-tuning

### [Tier 1] InstructGPT / RLHF (Ouyang et al., 2022)
**Link:** https://arxiv.org/abs/2203.02155

**Contribution:** Reinforcement Learning from Human Feedback (RLHF) pipeline: SFT → reward model training → PPO fine-tuning. Made GPT-3 helpful and less harmful without losing capability.

**Key result:** InstructGPT (1.3B) preferred over GPT-3 (175B) by human raters in 85% of comparisons. Alignment at a fraction of model size.

**The pipeline:**
```
1. SFT: fine-tune GPT-3 on ~13k human-written demonstrations
2. RM: train reward model on ~33k preference pairs (human A/B labels)
3. PPO: optimize GPT-3 against RM using PPO (KL-penalized from SFT policy)
```

---

### [Tier 1] DPO (Rafailov et al., 2023)
**Link:** https://arxiv.org/abs/2305.18290

**Contribution:** Direct Preference Optimization — eliminates the reward model and RL loop. Optimizes a closed-form objective directly from preference pairs.

**Key result:** Matches or exceeds PPO-RLHF quality on summarization and dialogue tasks with much simpler training (no RL, no separate RM).

**Core insight:**
```
L_DPO = -E[(y_w, y_l, x)] log σ(β log (π_θ(y_w|x)/π_ref(y_w|x)) 
                                  - β log (π_θ(y_l|x)/π_ref(y_l|x)))
```
Maximizes the gap in log-likelihood between the preferred and rejected response, relative to a reference policy.

---

### [Tier 2] Constitutional AI (Bai et al., Anthropic 2022)
**Link:** https://arxiv.org/abs/2212.08073

**Contribution:** Replace human harmlessness labels with AI-generated labels guided by a written "constitution." RLAIF scales alignment without proportional human annotation cost.

**Key result:** Claude models trained with CAI are rated more helpful AND less harmful than RLHF-only baselines.

---

### [Tier 2] GRPO / DeepSeek-R1 (DeepSeek, 2025)
**Link:** https://arxiv.org/abs/2501.12948

**Contribution:** Group Relative Policy Optimization eliminates the value network from PPO. Demonstrated that pure RL on verifiable rewards (math correctness, code test pass) produces o1-level reasoning in open-weight models.

**Key result:** DeepSeek-R1 matches OpenAI o1 on AIME 2024 (79.8% vs 79.2%) and exceeds it on Codeforces Elo (2029 vs 1891).

---

## Efficient Fine-Tuning (PEFT)

### [Tier 1] LoRA (Hu et al., 2021)
**Link:** https://arxiv.org/abs/2106.09685

**Contribution:** Low-Rank Adaptation — decompose weight updates into low-rank matrices `ΔW = BA`. Fine-tune only A and B (r×k + d×r parameters vs d×k for full FT).

**Key result:** Fine-tune GPT-3 175B with 10,000× fewer trainable parameters and no inference overhead (merge BA into W post-training). Within 1% of full fine-tuning on most benchmarks.

**Interview hook:** Why initialize B=0? So ΔW=BA=0 at initialization — the adapter starts as a no-op, preserving the pretrained model's behavior.

---

### [Tier 1] QLoRA (Dettmers et al., 2023)
**Link:** https://arxiv.org/abs/2305.14314

**Contribution:** Fine-tune large models on consumer hardware by quantizing the frozen base model to 4-bit (NF4) while keeping LoRA adapters in bf16.

**Key result:** Fine-tune a 65B model on a single 48GB GPU with only ~1-2% quality loss vs full bf16 fine-tuning.

**Three key innovations:**
1. NF4 (NormalFloat4): 4-bit dtype optimized for normally-distributed weights
2. Double quantization: quantize the quantization constants (~0.5 bit savings)
3. Paged optimizers: offload optimizer states to CPU RAM during gradient spikes

---

## Attention Efficiency

### [Tier 1] Flash Attention (Dao et al., 2022)
**Link:** https://arxiv.org/abs/2205.14135

**Contribution:** IO-aware exact attention — never materializes the N×N attention matrix in HBM. Tiles Q, K, V to compute in SRAM. O(N) memory vs O(N²), 2-4× faster on A100.

**Key result:** 15% end-to-end speedup for GPT-2 training; enables longer context windows that were previously OOM.

---

### [Tier 2] Flash Attention 2 (Dao, 2023)
**Link:** https://arxiv.org/abs/2307.08691

**Contribution:** Better work partitioning across GPU warps, sequence-dimension parallelism. ~2× speedup over FA1, ~73% A100 theoretical FLOP utilization.

---

### [Tier 2] Flash Attention 3 (Shah et al., 2024)
**Link:** https://arxiv.org/abs/2407.08608

**Contribution:** Hopper GPU (H100) optimization — exploits WGMMA instructions, overlaps softmax and GEMM via pipelining, async TMA data movement. ~1.5-2× over FA2 on H100.

---

### [Tier 1] PagedAttention / vLLM (Kwon et al., 2023)
**Link:** https://arxiv.org/abs/2309.06180

**Contribution:** OS-inspired virtual memory management for KV caches. Fixed-size pages eliminate fragmentation; enables KV cache sharing for parallel beam search. 24× higher throughput than HuggingFace Transformers.

---

### [Tier 2] Grouped Query Attention (Ainslie et al., 2023)
**Link:** https://arxiv.org/abs/2305.13245

**Contribution:** G groups share K/V heads among H query heads. KV cache reduced H/G×. Near-MHA quality at G=8 (used in LLaMA 3, Mistral, Gemma).

---

## Positional Encoding

### [Tier 1] RoPE (Su et al., 2021)
**Link:** https://arxiv.org/abs/2104.09864

**Contribution:** Rotary Position Embedding — encode position by rotating Q and K vectors. Dot product depends only on relative position, not absolute. Naturally extends to longer sequences.

**Used in:** LLaMA, Mistral, Qwen, DeepSeek, Gemma — essentially every modern open LLM.

---

### [Tier 2] YaRN (Peng et al., 2023)
**Link:** https://arxiv.org/abs/2309.00071

**Contribution:** Non-uniform RoPE scaling for context extension. High-frequency dimensions: no scaling. Low-frequency: interpolate. Temperature correction for attention entropy. Enables 16-32× context extension with minimal fine-tuning.

---

## Architecture Innovations

### [Tier 1] Mixture of Experts / Switch Transformer (Fedus et al., 2021)
**Link:** https://arxiv.org/abs/2101.03961

**Contribution:** Scaled MoE to 1.6T parameters with top-1 routing. Matched T5-11B at 7× fewer FLOPs per step. Load balancing auxiliary loss prevents expert collapse.

---

### [Tier 1] Mamba (Gu & Dao, 2023)
**Link:** https://arxiv.org/abs/2312.00752

**Contribution:** Selective state space model — input-dependent B, C, Δ parameters allow content-based selection (like attention) with linear inference complexity. Custom CUDA parallel scan kernel achieves 5× throughput vs Transformers at 2K sequence length.

---

## Reasoning & Prompting

### [Tier 1] Chain-of-Thought Prompting (Wei et al., 2022)
**Link:** https://arxiv.org/abs/2201.11903

**Contribution:** Eliciting intermediate reasoning steps dramatically improves performance on multi-step reasoning tasks. Few-shot CoT: provide examples with step-by-step solutions. Zero-shot CoT: "Let's think step by step."

**Key result:** PaLM 540B + CoT: 56.9% GSM8K vs 17.9% without CoT. CoT is most effective above ~100B parameters.

---

### [Tier 2] Self-Consistency (Wang et al., 2022)
**Link:** https://arxiv.org/abs/2203.11171

**Contribution:** Sample multiple diverse reasoning paths, take the majority vote. Replaces greedy decoding with a self-consistency ensemble — no external verifier needed.

**Key result:** +17.9% on GSM8K vs single-chain CoT.

---

### [Tier 2] Tree of Thoughts (Yao et al., 2023)
**Link:** https://arxiv.org/abs/2305.10601

**Contribution:** Extend CoT from a chain to a tree — explore multiple reasoning paths, evaluate intermediate steps, backtrack when needed. Enables deliberate problem-solving for tasks requiring lookahead.

---

## Interpretability

### [Tier 2] Induction Heads (Olsson et al., 2022)
**Link:** https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html

**Contribution:** Identified the "induction head" circuit — a two-head attention mechanism responsible for in-context learning. Head 1 attends to the previous token; Head 2 attends to the token after the previous occurrence of the current token. Ablating induction heads dramatically reduces in-context learning performance.

---

### [Tier 2] Toy Models of Superposition (Elhage et al., Anthropic 2022)
**Link:** https://transformer-circuits.pub/2022/toy_model/index.html

**Contribution:** Neural networks with d dimensions can represent n > d features by encoding them as nearly-orthogonal vectors (superposition). Features are compressed because natural data is sparse. This explains why individual neurons are polysemantic and why direct neuron inspection is misleading.

---

### [Tier 2] Scaling & Evaluating Sparse Autoencoders (Anthropic 2023-24)
**Link:** https://transformer-circuits.pub/2023/monosemantic-features/index.html

**Contribution:** Train sparse autoencoders on model activations to decompose superposed representations into (near-)monosemantic features. Each SAE feature corresponds to an interpretable concept. First systematic method to "see inside" a large transformer at scale.

---

## Quick Reference Table

| Paper | Year | Key Concept | Tier |
|-------|------|-------------|------|
| Attention is All You Need | 2017 | Transformer architecture | 1 |
| BERT | 2018 | Bidirectional pre-training | 1 |
| GPT-3 | 2020 | In-context learning at scale | 1 |
| Scaling Laws (Kaplan) | 2020 | Power law scaling | 1 |
| InstructGPT/RLHF | 2022 | Human preference alignment | 1 |
| Chinchilla | 2022 | Compute-optimal training | 1 |
| Chain-of-Thought | 2022 | Reasoning via intermediate steps | 1 |
| LoRA | 2021 | Low-rank weight adaptation | 1 |
| Flash Attention | 2022 | IO-aware exact attention | 1 |
| RoPE | 2021 | Rotary position encoding | 1 |
| Switch Transformer (MoE) | 2021 | Sparse expert routing | 1 |
| Mamba | 2023 | Selective state spaces | 1 |
| QLoRA | 2023 | 4-bit quantized fine-tuning | 1 |
| DPO | 2023 | RL-free preference learning | 1 |
| PagedAttention/vLLM | 2023 | KV cache virtual memory | 1 |
| Constitutional AI | 2022 | AI-feedback alignment | 2 |
| Self-Consistency | 2022 | Majority vote CoT | 2 |
| GQA | 2023 | Grouped KV heads | 2 |
| YaRN | 2023 | RoPE context extension | 2 |
| Flash Attention 2 | 2023 | Warp-level optimization | 2 |
| Tree of Thoughts | 2023 | Tree-structured reasoning | 2 |
| Induction Heads | 2022 | ICL circuit identification | 2 |
| Superposition | 2022 | Feature compression theory | 2 |
| Sparse Autoencoders | 2023 | Monosemantic interpretability | 2 |
| DeepSeek-R1/GRPO | 2025 | RL reasoning without value net | 2 |

---

*Last updated: May 2026 | 25 papers, tiered by interview importance*

## Flashcards

**β log (π_θ(y_l|x)/π_ref(y_l|x)))?** #flashcard
β log (π_θ(y_l|x)/π_ref(y_l|x)))
