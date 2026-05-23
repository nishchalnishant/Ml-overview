---
module: Emerging Topics
topic: Emerging Trends
subtopic: Mixture Of Experts
status: unread
tags: [emergingtopics, ml, emerging-trends-mixture-of-exp]
---
# Mixture of Experts (MoE)

How MoE enables training trillion-parameter models while keeping inference cost at a fraction of the full parameter count. The architecture behind DeepSeek-V3, Mixtral, GPT-4 (rumored), and Gemini 1.5.

---

## 1. The Core Problem MoE Solves

A dense transformer with 70B parameters activates all 70B parameters for every token. Doubling model capacity to 140B doubles both memory AND compute per token — they scale together.

**MoE breaks this coupling.** A 660B MoE model with 8 active experts per token uses the same FLOPs per token as a 70B dense model — but the model has access to 660B parameters of knowledge. Compute scales with active parameters; capacity scales with total parameters.

```
Dense 70B:  70B params activated per token  = 70B FLOPs/token
MoE  660B:  8/64 experts activated         = ~70B FLOPs/token  (same compute)
             but 660B total parameters      = 9.4× more capacity
```

This is why every frontier model at scale (GPT-4, Gemini 1.5, Mixtral, DeepSeek) uses MoE.

---

## 2. Architecture

### Standard MoE Layer

Each transformer block has an FFN sublayer. MoE replaces the single FFN with N expert FFNs and a learned router that selects K of them per token.

```
Standard FFN:
  h = FFN(x) = W_2 · SiLU(W_1 · x)

MoE FFN (N experts, top-K routing):
  gates = softmax(x · W_router)              # [batch, seq, N] — score for each expert
  top_k_gates, top_k_idx = topk(gates, K)    # select K highest-scoring experts
  h = Σ_{i ∈ top_k} gate_i · Expert_i(x)    # weighted sum of K expert outputs
```

Each expert_i is a full FFN with its own W_1, W_2 weights. The router W_router is a learned projection (d_model × N).

### Typical Configuration

| Model | Total params | Active params | Experts | Top-K | Expert intermediate dim |
|---|---|---|---|---|---|
| Mixtral 8×7B | 46.7B | 12.9B | 8 | 2 | Same as 7B dense |
| Mixtral 8×22B | 140.6B | 39.1B | 8 | 2 | Same as 22B dense |
| DeepSeek-V3 | 671B | 37B | 256 | 8 | Smaller (fine-grained) |
| Grok-1 | 314B | 86B | 8 | 2 | — |
| Gemini 1.5 Pro | ~1T (est.) | ~50B (est.) | many | — | — |

**Fine-grained experts (DeepSeek-V3 approach)**: instead of 8 large experts, use 256 small experts. This increases routing flexibility — the router can compose more specialized combinations — and allows higher diversity in what each token activates.

---

## 3. The Load Balancing Problem

**Without load balancing**: the router collapses — it learns to send all tokens to the same 1-2 experts (these experts get the most gradient signal and improve fastest, making the router prefer them even more). Most experts receive no training signal and become useless.

**Auxiliary load balancing loss** (Mixtral/standard approach):

$$L_{aux} = \alpha \cdot N \cdot \sum_{i=1}^{N} f_i \cdot P_i$$

where:
- $f_i$ = fraction of tokens routed to expert $i$ (computed without gradients)
- $P_i$ = average router probability for expert $i$ (computed with gradients)
- $\alpha$ = loss coefficient (typically 0.01)
- The loss is minimized when $f_i = 1/N$ for all $i$ (uniform load)

**DeepSeek-V3 approach — auxiliary-loss-free load balancing**: instead of a loss term, add a per-expert bias term $b_i$ to router logits. If expert $i$ is overloaded, $b_i$ decreases, making the router less likely to select it. This decouples routing quality from load balancing, improving both.

```python
# Router with bias (DeepSeek-V3 style)
def route(x, W_router, expert_biases):
    logits = x @ W_router.T
    logits_for_routing = logits + expert_biases  # add per-expert bias
    top_k_gates, top_k_idx = torch.topk(logits_for_routing, K)
    # But use original logits for gate weights (not biased)
    actual_gates = softmax(torch.gather(logits, -1, top_k_idx))
    return top_k_gates, top_k_idx, actual_gates
```

---

## 4. Expert Parallelism

In distributed training/inference, different experts live on different GPUs. This is **expert parallelism**, which is distinct from tensor parallelism or pipeline parallelism.

```
All-to-All communication pattern:

GPU 0 (experts 0,1):    receives tokens routed to experts 0,1 from all GPUs
GPU 1 (experts 2,3):    receives tokens routed to experts 2,3 from all GPUs
GPU 2 (experts 4,5):    receives tokens routed to experts 4,5 from all GPUs
GPU 3 (experts 6,7):    receives tokens routed to experts 6,7 from all GPUs

Step 1: All-to-All scatter  — each GPU sends its tokens to the correct expert GPU
Step 2: Expert computation  — each GPU runs its 2 expert FFNs
Step 3: All-to-All gather   — results scatter back to original GPUs
```

The all-to-all communication is the bottleneck — at scale, this requires high-bandwidth interconnects (NVLink, InfiniBand). DeepSeek-V3 was trained on H800 GPUs (NVLink within nodes) with careful overlap of compute and communication.

---

## 5. MoE at Inference

**Memory**: MoE requires loading ALL expert weights into memory at inference time (you don't know in advance which experts will be activated). A 660B MoE model needs 660B × 2 bytes = 1.32 TB of GPU memory just for weights. This requires many GPUs even though compute per token is manageable.

**Expert offloading** (for consumer hardware): keep frequently-used experts in GPU memory, evict cold experts to CPU RAM. Load expert weights just-in-time when the router selects them. Throughput drops significantly but allows running large MoE on limited GPU memory.

**Activation patterns**: experts specialize during training. Studies on Mixtral show experts develop domain preferences (certain experts activate more for code, others for math, others for French). This specialization is emergent — not explicitly trained.

---

## 6. MoE vs Dense Trade-offs

| Dimension | Dense | MoE |
|---|---|---|
| FLOPs per token | Proportional to total params | Proportional to active params |
| Memory at inference | Proportional to total params | Proportional to total params |
| Training efficiency | Simpler | Requires load balancing, all-to-all comms |
| Serving complexity | Simple | Needs expert routing infrastructure |
| Quality per FLOP | Baseline | Better (more capacity same compute) |
| Long tail knowledge | Worse (params spread thinly) | Better (specialists) |
| Latency | Predictable | Variable (depends on routing) |

**When dense wins**: small models (<13B), latency-critical serving, hardware without fast interconnects, fine-tuning (load imbalance is harder to control on small datasets).

---

## 7. Shared Expert (DeepSeek-V3 Innovation)

DeepSeek-V3 uses a hybrid: N_shared always-active experts + N_routed selectively-activated experts.

```
Output = FFN_shared_1(x) + FFN_shared_2(x) + Σ_{i ∈ top_K_routed} gate_i · Expert_i(x)
```

The shared experts capture universal language knowledge (always active), while routed experts handle specialization. This prevents a failure mode where the router selects redundant general-purpose experts instead of specialized ones.

---

## 8. Soft MoE

Standard MoE uses hard top-K routing — tokens are assigned discretely to K experts. The argmax is not differentiable, creating training instability.

**Soft MoE** (Google, 2023): assign each token a weighted combination across ALL experts (weighted by softmax gates). Fully differentiable. Less efficient (every expert processes every token) but more stable training. Used in some research models but not widely in production due to compute cost.

---

## Canonical Interview Q&As

**Q: Explain the MoE routing mechanism and the load balancing problem. How does DeepSeek-V3 solve it differently?**
A: The router is a linear projection W_router (d_model × N) that maps each token's hidden state to N expert scores, then softmax + top-K selects the K highest-scoring experts. The fundamental problem: without constraints, the router collapses to always selecting the same 1-2 experts. These experts receive the most gradient updates and improve fastest, making the router prefer them even more — a rich-get-richer spiral where most experts never get trained. Standard fix: add auxiliary loss L_aux = α·N·Σf_i·P_i that penalizes unequal load distribution (f_i = fraction of tokens to expert i, P_i = router probability for expert i). When load is uniform, f_i = P_i = 1/N minimizes the loss. DeepSeek-V3 instead adds a per-expert bias b_i to router logits. Overloaded experts get decreasing b_i, naturally reducing their selection probability without a loss term. This allows routing quality (gate values used for weighted sum) to be decoupled from load balancing (biased logits used for selection) — the key insight is that selection and weighting can use different scores.

**Q: Why does a 660B MoE model use the same FLOPs per token as a 70B dense model, yet outperform it?**
A: In a top-2 MoE with 16 experts, each token activates 2 expert FFNs. If each expert has the same intermediate dimension as a 35B-equivalent FFN, the total FLOPs per token = 2 experts × 35B-equivalent = 70B-equivalent. But the model stores 16 × 35B = 560B parameters. The quality advantage comes from capacity: different tokens can be routed to different expert combinations, effectively giving the model 16 specialized sub-networks. A dense 70B model must use the same 70B parameters to be good at math, code, French, and biology simultaneously. A 560B MoE can have dedicated experts for each domain, with no competition for parameter space. Empirically, MoE models achieve the quality of a 3-4× larger dense model for the same training FLOPs (Mixtral 8×7B matches or beats LLaMA-2 70B at 12.9B active parameters).

**Q: What are the inference infrastructure challenges for large MoE models?**
A: Two distinct problems: (1) Memory: all expert weights must be loaded to GPU memory even though only 2/64 are used per token. DeepSeek-V3 at 671B total params requires 671B × 2 bytes (bf16) = 1.34 TB of GPU memory — that's 17 H100-80GB GPUs just for weights. This makes large MoE models expensive to deploy even though per-token compute is cheap. (2) Latency vs throughput: at small batch sizes (single user), per-token latency is fine but most experts are idle (waste). At large batch sizes, tokens spread across more experts (better utilization) but require all-to-all communication overhead between expert-parallel GPUs. Solutions: expert offloading to CPU for memory reduction (hurts latency), speculative expert pre-loading based on predicted routing, and dense distillation — train a smaller dense model to mimic the MoE, then serve the smaller model.

## Flashcards

**$f_i$ = fraction of tokens routed to expert $i$ (computed without gradients)?** #flashcard
$f_i$ = fraction of tokens routed to expert $i$ (computed without gradients)

**$P_i$ = average router probability for expert $i$ (computed with gradients)?** #flashcard
$P_i$ = average router probability for expert $i$ (computed with gradients)

**$\alpha$ = loss coefficient (typically 0.01)?** #flashcard
$\alpha$ = loss coefficient (typically 0.01)

**The loss is minimized when $f_i = 1/N$ for all $i$ (uniform load)?** #flashcard
The loss is minimized when $f_i = 1/N$ for all $i$ (uniform load)
