---
module: LLMs
topic: Revision Card
subtopic: ""
status: unread
tags: [llms, revision, cheatsheet]
---
# LLMs — 10-Minute Revision Card

---

## Mental Model: LLM = Compressed World Model

Pretraining on next-token prediction forces the model to learn syntax, semantics, facts, and reasoning as a byproduct. It's a capability installer, not a values installer — alignment is a separate step.

---

## Architecture at a Glance (Decoder-Only Transformer)

```
Input tokens → Embedding + Positional Encoding
    ↓ × N layers
[Masked Self-Attention] → [Add & Norm] → [FFN] → [Add & Norm]
    ↓
Linear → Softmax → Next-token probability distribution
```

**Key numbers (LLaMA 3 70B):** 80 layers, 8192 context, GQA (8 KV heads), 15T training tokens

**Where the parameters live:**
- ~2/3 in FFN layers (knowledge storage)
- ~1/3 in attention (routing / pattern matching)

---

## Attention — Fast Reference

$$\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

| Variant | KV Heads | Memory | Quality |
|---------|----------|--------|---------|
| MHA | = Q heads (H) | Largest | Best |
| GQA | H/G (e.g. 8 vs 64) | H/G× smaller | Near-MHA |
| MQA | 1 | 1/H× of MHA | Slightly lower |

**KV cache per token per layer:** $2 \times n_{kv\_heads} \times d_{head} \times \text{bytes}$

**Why GQA matters:** LLaMA 3 70B, 4096 context, batch 32 → full MHA KV cache ≈ 4 GB. GQA (8 KV heads vs 64) reduces this 8×.

---

## Training Stages

| Stage | Objective | Data | What it installs |
|-------|-----------|------|-----------------|
| Pretraining | Next-token prediction | Trillions of tokens | Language, facts, reasoning |
| SFT | Supervised fine-tune on demos | Curated instruction pairs | Instruction following |
| RLHF | RM + PPO | Human preference pairs | Human preference alignment |
| DPO | Direct preference opt. | Preference pairs | Alignment (no RL, 2 models) |
| ORPO | SFT + odds ratio in one loss | Preference pairs | Alignment (no ref model, 1 model) |

**LM Loss:** $\mathcal{L} = -\frac{1}{T}\sum_{t=1}^{T} \log P_\theta(x_t \mid x_1,\ldots,x_{t-1})$

**Token efficiency:** LLaMA 3 trained 70B params on 15T tokens (~214 tokens/param) — deliberately overtrained for inference efficiency (cheaper to serve).

---

## RLHF vs DPO

**RLHF:**
1. Train reward model (RM) on human preference pairs
2. Use PPO to optimize policy toward RM reward
3. KL penalty prevents policy from drifting too far from SFT

**DPO:** skip the RM entirely — directly optimize log ratio of preferred over rejected:
$$\mathcal{L}_{\text{DPO}} = -\log\sigma\!\left(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta\log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)$$

**When DPO wins:** simpler pipeline, no reward hacking risk, stable training. **When RLHF wins:** complex reward signals, online learning from real feedback. **When ORPO wins:** memory-constrained (no reference model), high-quality data, want SFT + alignment in one pass.

---

## Tokenization

- **BPE (Byte-Pair Encoding):** merge most frequent byte pairs iteratively → variable-length tokens. Used by GPT-4, LLaMA.
- **WordPiece:** similar but merges maximize likelihood. Used by BERT.
- **SentencePiece:** language-agnostic BPE operating on raw text (no pre-tokenization).

**Gotchas:**
- Numbers tokenize poorly — "1234567" may be 3-4 tokens; arithmetic is hard partly because of this
- Rare words split into many subwords — longer context needed
- Different tokenizers → different vocabularies → models not interchangeable

---

## KV Cache

**Purpose:** during autoregressive generation, reuse K and V from previous tokens instead of recomputing.

**Memory:** grows linearly with sequence length and batch size.

**Bottleneck:** at long context (>4K tokens) or large batch, KV cache dominates GPU memory.

**Solutions:** GQA/MQA (reduce KV heads), sliding window attention (fixed cache window), paged attention (vLLM — store cache in non-contiguous pages like virtual memory).

---

## Fine-Tuning Methods

| Method | Trains | Memory | Use when |
|--------|--------|--------|---------|
| Full fine-tune | All params | High | Enough data + compute |
| LoRA | Low-rank adapters only | Low | Limited compute / many tasks |
| QLoRA | LoRA on quantized base | Very low | Consumer GPU |
| Prefix tuning | Soft prompts (KV at each layer) | Very low | Inference-time task switching |
| Adapters | Bottleneck MLP per layer | Low | Multi-task, nonlinear adaptation |

**LoRA:** $W = W_0 + AB$ where $A \in \mathbb{R}^{d \times r}$, $B \in \mathbb{R}^{r \times k}$, rank $r \ll \min(d,k)$.
Typical $r=8$ or $r=16$; trains <1% of parameters.

---

## Inference Optimization

| Technique | What it does | Speedup |
|-----------|-------------|---------|
| Quantization (INT8/INT4) | Reduce weight precision | 2-4× memory, 1.5-2× speed |
| Flash Attention | Fused kernel, IO-optimal attention | 2-4× for long context |
| Speculative Decoding | Draft model proposes, target verifies | 2-3× latency |
| Continuous Batching | Dynamic request batching | 2-10× throughput |
| Paged Attention | vLLM's virtual KV cache paging | Near-optimal GPU utilization |

**Speculative decoding intuition:** small fast model proposes k tokens; large model verifies in one forward pass (parallel). If accepted, k tokens for ~1 forward pass cost.

---

## Scaling Laws (Chinchilla)

**Optimal compute allocation:** $N_{params} \approx \frac{C}{6 \times D_{tokens}}$

**Chinchilla rule:** ~20 tokens per parameter for compute-optimal training.

**Key insight:** GPT-3 (175B, 300B tokens) was severely undertrained. Chinchilla (70B, 1.4T tokens) matched GPT-3 quality at 4× fewer params.

**Modern practice:** deliberately overtrain smaller models (LLaMA, Mistral) for cheaper inference.

---

## RAG vs Fine-Tuning

| Dimension | RAG | Fine-Tuning |
|-----------|-----|-------------|
| Knowledge update | Real-time (retrieval) | Requires retraining |
| Factual grounding | Source-cited, verifiable | Baked into weights |
| Compute at inference | Higher (retrieval + generation) | Standard |
| Best for | Changing facts, private data | Style/behavior change |

**When RAG fails:** retrieval quality is the bottleneck. Bad chunks → hallucinated synthesis. Fix: better chunking, hybrid search (BM25 + dense), reranker.

---

## Hallucination Sources

1. **Training data:** conflicting facts, low-frequency entities memorized poorly
2. **Decoding:** high temperature → tail of distribution → unlikely tokens
3. **Prompt:** model sycophantically agrees with false premises in question
4. **RAG:** retrieved irrelevant context → model synthesizes confident fiction

**Mitigations:** retrieval grounding, citation, chain-of-thought (reasoning makes mistakes detectable), temperature reduction, RLHF on factuality.

---

## Mixture of Experts (MoE)

- Only a fraction (top-K) of experts activate per token → same quality, fraction of compute
- **LLaMA MoE / Mixtral:** 8 experts, top-2 activated per token → ~25% active params
- **Load balancing loss:** prevent all tokens routing to same expert
- **Gotcha:** total parameters large (memory), active params small (compute). Memory-bound, not compute-bound.

---

## Interview Quick-Draws

**"How does attention scale with context length?"**
→ O(T²) in memory and compute — quadratic. At T=32K, attention matrix = 4GB per layer. Mitigations: Flash Attention (IO efficiency), sliding window, GQA (KV cache), sparse attention.

**"RLHF vs DPO?"**
→ RLHF: train reward model, then PPO loop — complex but flexible. DPO: directly optimize preference pairs without RM or RL — simpler, no reward hacking, often matches RLHF quality.

**"What's a KV cache?"**
→ Store computed K and V tensors from previous tokens so autoregressive generation doesn't recompute them. Grows linearly with context. GQA reduces size by sharing KV heads across query groups.

**"Why does temperature matter?"**
→ Scales logits before softmax: $p_i \propto e^{z_i/T}$. Low T → peaked distribution → deterministic. High T → flat distribution → creative but hallucinates. T=0 ≈ greedy decoding.

**"What's the difference between SFT and RLHF?"**
→ SFT: supervised on human demonstrations — teaches format and basic instruction following. RLHF: optimize reward model trained on comparisons — aligns to human preferences (helpfulness, safety). RLHF builds on SFT.

**"How does LoRA work?"**
→ Freeze pretrained weights $W_0$. Learn low-rank update $\Delta W = AB$ where $r \ll d$. Merge at inference: $W = W_0 + AB$. Reduces trainable params from $d^2$ to $2dr$ — typically 100-1000× fewer.

**"What's ORPO and when do you use it over DPO?"**
→ ORPO (Odds Ratio Preference Optimization) combines SFT and preference alignment in one loss: $\mathcal{L}_{ORPO} = \mathcal{L}_{SFT} + \lambda \cdot \mathcal{L}_{OR}$. No reference model needed (DPO needs 2 models loaded; ORPO needs 1). Use ORPO when memory-constrained or preference data is clean and high-quality. Use DPO when data is noisier (KL anchor helps) or you have a separate high-quality SFT model already.
