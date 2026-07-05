---
module: Llms
topic: Moe Advanced And Routing
subtopic: ""
status: unread
tags: [llms, ml, moe-advanced-and-routing]
---
# Mixture of Experts (MoE) — Advanced and Routing

MoE is the architecture behind Mixtral, DeepSeek, Qwen3, and GPT-4 (rumored). Understanding routing, load balancing, and failure modes is essential for 2024/2025 LLM interviews.

> For the basic "how does a dense FFN become a sparse MoE FFN" architectural sketch inside the standard Transformer block, see [03-deep-learning/components/11-transformers.md](../03-deep-learning/components/11-transformers.md#mixture-of-experts-moe-ffn). This file goes deeper: router variants, load-balancing loss derivations, capacity factors, expert parallelism, and production models.

---

## 1. Core MoE Architecture

**Standard dense Transformer FFN:**
$$\text{FFN}(x) = W_2 \cdot \text{ReLU}(W_1 x)$$

**MoE FFN:** Replace with N expert FFNs, activate only top-K per token:
$$\text{MoE}(x) = \sum_{i \in \text{TopK}(G(x))} G_i(x) \cdot E_i(x)$$

where $G(x) \in \mathbb{R}^N$ are gating scores and $E_i(x)$ is the i-th expert FFN.

**Sparse activation:** If N=64 experts and K=2, each token activates 2/64 = 3.1% of expert parameters. Total parameters scale with N, but FLOPs per token stay fixed at 2× dense equivalent.

```
Dense 7B model:         7B params, 7B FLOPs/token
MoE 7B active (46B total): 46B params, 7B FLOPs/token  ← Mixtral 8×7B
```

**FLOPs vs parameters distinction:**
- **Parameters:** determine model capacity and memory footprint
- **Active FLOPs:** determine compute cost per token
- MoE decouples these — large capacity at dense-model inference cost

---

## 2. Router / Gating Mechanism

### Softmax Router (Standard)

$$G(x) = \text{Softmax}(x W_g)$$
$$\text{TopK}(G(x)) = \text{indices of top K values}$$

**Final gating scores (normalized over selected experts):**
$$\tilde{G}_i(x) = \frac{G_i(x)}{\sum_{j \in \text{TopK}} G_j(x)}$$

```python
class MoELayer(nn.Module):
    def __init__(self, d_model, n_experts, top_k, expert_dim):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.gate = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, expert_dim),
                nn.SiLU(),
                nn.Linear(expert_dim, d_model)
            ) for _ in range(n_experts)
        ])
    
    def forward(self, x):
        # x: [batch, seq, d_model]
        B, S, D = x.shape
        x_flat = x.view(-1, D)  # [B*S, D]
        
        # Gating
        logits = self.gate(x_flat)                          # [B*S, N]
        scores = F.softmax(logits, dim=-1)
        top_k_scores, top_k_indices = torch.topk(scores, self.top_k, dim=-1)
        top_k_scores = top_k_scores / top_k_scores.sum(dim=-1, keepdim=True)
        
        # Expert dispatch
        output = torch.zeros_like(x_flat)
        for k in range(self.top_k):
            expert_idx = top_k_indices[:, k]   # which expert each token goes to
            weight = top_k_scores[:, k]         # gating weight
            
            for i in range(self.n_experts):
                mask = (expert_idx == i)
                if mask.any():
                    output[mask] += weight[mask].unsqueeze(-1) * self.experts[i](x_flat[mask])
        
        return output.view(B, S, D)
```

### Expert Choice Routing (Google, 2022)

Instead of tokens choosing experts, **experts choose tokens**.

Each expert selects its top-C tokens from the batch:
$$\text{Expert}_i \text{ selects top-C tokens by } G_i(x_j) \text{ over all tokens } j$$

**Advantage:** Perfect load balance by construction (every expert processes exactly C tokens).  
**Disadvantage:** A token may be processed by 0 experts (dropped) or many experts — inference is harder.

---

## 3. Load Balancing Loss

**Problem:** Without regularization, the router collapses — routes all tokens to 1–2 experts (popularity bias). Remaining experts receive no gradient, atrophy.

**Expert collapse:** Router learns that expert 0 is slightly better → sends more tokens to expert 0 → expert 0 gets more gradient, improves further → feedback loop → all tokens go to expert 0.

### Auxiliary Load-Balancing Loss (Switch Transformer)

$$\mathcal{L}_{balance} = \alpha \cdot N \cdot \sum_{i=1}^N f_i \cdot P_i$$

where:
- $f_i = \frac{\text{tokens dispatched to expert } i}{\text{total tokens}}$ (fraction of tokens)
- $P_i = \frac{1}{T}\sum_{x} G_i(x)$ (mean routing probability for expert i)
- N = number of experts
- α = load balancing coefficient (typically 0.01–0.1)

**Why f_i × P_i?** f_i is non-differentiable (argmax operation); P_i provides the gradient signal. Their product pushes the router toward uniform expert utilization.

**Total loss:**
$$\mathcal{L}_{total} = \mathcal{L}_{LM} + \alpha \cdot \mathcal{L}_{balance}$$

```python
def load_balance_loss(router_probs, expert_indices, n_experts, alpha=0.01):
    """
    router_probs: [n_tokens, n_experts] — softmax output
    expert_indices: [n_tokens, top_k] — selected experts per token
    """
    n_tokens = router_probs.shape[0]
    
    # f_i: fraction of tokens sent to each expert
    token_counts = torch.zeros(n_experts, device=router_probs.device)
    for k in range(expert_indices.shape[1]):
        token_counts.scatter_add_(0, expert_indices[:, k], 
                                   torch.ones(n_tokens, device=router_probs.device))
    f = token_counts / (n_tokens * expert_indices.shape[1])
    
    # P_i: mean routing probability per expert
    P = router_probs.mean(dim=0)
    
    return alpha * n_experts * (f * P).sum()
```

### DeepSeek MoE: Fine-grained Expert Segmentation

DeepSeek-V2 splits experts into:
- **Shared experts** (always active): handle common knowledge
- **Routed experts** (top-K selected): handle specialized knowledge

$$\text{MoE}(x) = \sum_{i=1}^{K_s} E_i^{shared}(x) + \sum_{i \in \text{TopK}(G(x))} \tilde{G}_i(x) \cdot E_i^{routed}(x)$$

**Advantage:** Shared experts prevent knowledge redundancy across routed experts. Routed experts can specialize further.

---

## 4. Token Dropping and Capacity Factor

**Expert capacity:** Each expert has a fixed buffer size (capacity). If more tokens are routed to an expert than its capacity, excess tokens are dropped.

$$\text{Expert capacity} = \left\lfloor \frac{\text{tokens per batch} \times \text{capacity factor}}{N} \right\rfloor$$

With capacity_factor=1.25, each expert can handle 25% more than the average load.

**Trade-off:**
- capacity_factor ↑ → fewer dropped tokens, more memory
- capacity_factor ↓ → more dropped tokens (information loss), less memory

**Training vs inference:**
- Training: drop tokens (ignore their gradients) at overloaded experts
- Inference: typically use capacity_factor=2.0 to minimize drops, or use "no drop" mode

```python
def dispatch_with_capacity(tokens, expert_indices, n_experts, capacity_factor=1.25):
    """Dispatch tokens to experts respecting capacity limits."""
    n_tokens = tokens.shape[0]
    capacity = int(n_tokens * capacity_factor / n_experts)
    
    expert_counts = torch.zeros(n_experts, dtype=torch.long)
    dispatch_mask = torch.zeros(n_tokens, dtype=torch.bool)
    
    for i, expert_id in enumerate(expert_indices):
        if expert_counts[expert_id] < capacity:
            dispatch_mask[i] = True
            expert_counts[expert_id] += 1
        # else: token i is dropped
    
    dropped_fraction = (~dispatch_mask).float().mean()
    return dispatch_mask, dropped_fraction
```

---

## 5. Expert Parallelism

With N=64 experts across 64 GPUs, each GPU holds 1 expert. Routing requires:
1. **All-to-All communication:** send each token to its assigned expert's GPU
2. **Expert computation:** local FFN on each GPU
3. **All-to-All communication:** return computed results to origin GPUs

**Communication cost per MoE layer:**
$$2 \times \text{all-to-all}(B \times S \times d_{model} \times \text{dtype\_bytes})$$

This is the main overhead of MoE vs dense models in distributed training.

**Expert parallelism + TP + DP:**
```
Expert parallel (EP=8): each group of 8 GPUs holds 1 expert per MoE layer
Tensor parallel (TP=8): within each expert group, split expert FFN across 8 GPUs
Data parallel (DP=N): multiple replicas
```

---

## 6. Expert Specialization (What Do Experts Learn?)

Empirically, experts develop semantic specialization:
- Some experts activate on specific syntactic patterns (punctuation, numbers)
- Some on semantic domains (medical text, code, math)
- Some on languages (English expert vs Chinese expert)

**Checking expert specialization:**
```python
def analyze_expert_routing(model, dataset, n_experts):
    """Track which tokens go to which experts."""
    expert_token_counts = defaultdict(Counter)
    
    for batch in dataset:
        with torch.no_grad():
            _, routing_info = model(batch, return_routing=True)
        
        for layer_idx, (indices, tokens) in enumerate(routing_info):
            for token, expert in zip(tokens, indices):
                expert_token_counts[layer_idx][expert] += 1
    
    # Compute entropy of routing distribution (low entropy = collapsed)
    for layer, counts in expert_token_counts.items():
        total = sum(counts.values())
        probs = [c/total for c in counts.values()]
        entropy = -sum(p * math.log(p + 1e-10) for p in probs)
        max_entropy = math.log(n_experts)
        print(f"Layer {layer}: routing entropy = {entropy:.2f}/{max_entropy:.2f}")
```

Healthy MoE: routing entropy close to max_entropy (uniform). Expert collapse: entropy near 0.

---

## 7. MoE Models in Production

| Model | N experts | Top-K | Active params | Total params |
|---|---|---|---|---|
| Mixtral 8×7B | 8 | 2 | 12.9B | 46.7B |
| Mixtral 8×22B | 8 | 2 | 39B | 141B |
| DeepSeek-V2 | 160 | 6 | 21B | 236B |
| DeepSeek-V3 | 256 | 8 | 37B | 671B |
| Qwen3 MoE | 128 | 8 | 22B | 235B |
| GPT-4 (rumored) | ~16 | 2 | ~111B | ~1.8T |

**Mixtral routing:** Standard top-2 softmax, no shared experts, auxiliary load balance loss.

**DeepSeek routing:** Fine-grained (160 experts), shared+routed split, expert-level balance loss + device-level balance loss for EP efficiency.

---

## 8. iRoPE (Qwen3)

Qwen3 introduces **iRoPE**: every 4th Transformer layer uses no positional encoding (NoPE).

**Motivation for MoE context:** With many layers and large context, positional encoding can become a bottleneck for routing decisions — tokens at similar positions may be routed similarly. NoPE layers break this positional dependency, enabling position-agnostic routing that focuses on semantic content.

```
Layer 0: RoPE + MoE
Layer 1: RoPE + MoE
Layer 2: RoPE + MoE
Layer 3: NoPE + MoE  ← position-agnostic attention
Layer 4: RoPE + MoE
...
```

**Result:** Better extrapolation to contexts beyond training length (enables 10M+ context).

---

## 9. Failure Modes and Mitigations

| Failure | Symptom | Cause | Fix |
|---|---|---|---|
| Expert collapse | 1–2 experts get all traffic | No load balance loss | Add α · L_balance |
| Token dropping | High dropped token % | capacity_factor too low | Increase capacity_factor or use no-drop |
| Routing instability | Expert assignments flip between steps | Learning rate too high | LR warmup, lower LR for router |
| Expert redundancy | Multiple experts learn same function | No diversity incentive | Orthogonality regularization |
| All-to-All bottleneck | MoE layer much slower than dense | EP communication overhead | Overlap communication with computation |

---

## Canonical Interview Q&As

**Q: What is the FLOPs vs parameter trade-off in MoE models?**  
A: MoE decouples model capacity (parameters) from compute cost (FLOPs). Mixtral 8×7B has 46.7B total parameters but only 12.9B are active per token (top-2 of 8 experts). Training and inference FLOPs are close to a 13B dense model, but the model has the capacity of a 46.7B model — because different experts can specialize. The catch: all 46.7B parameters must reside in memory, so serving Mixtral 8×7B requires ~93GB (fp16), similar to a 47B dense model despite dense-equivalent compute.

**Q: Why does the router collapse and how does the load-balancing loss fix it?**  
A: Without regularization, the router has a feedback loop: a slightly better expert receives more tokens → better gradient signal → improves further → receives even more tokens. Eventually all tokens route to 1–2 experts (collapse). The load-balancing loss L_balance = α·N·Σ f_i·P_i penalizes uneven routing. f_i (token fraction) is non-differentiable, but P_i (mean routing probability) provides the gradient. When expert i is overloaded (high f_i), the loss increases proportionally to P_i, pushing the router to reduce routing probability to that expert. α = 0.01 is typical — too high breaks task performance, too low doesn't prevent collapse.

**Q: How would you debug an MoE model that shows degraded performance after 50K training steps?**  
A: Check routing entropy at each MoE layer — if entropy dropped significantly, expert collapse is happening. Also monitor: (1) per-expert token fraction distribution (should be roughly uniform); (2) dropped token percentage (should be <5%); (3) load balance loss magnitude; (4) gradient norm per expert (dead experts have near-zero gradient). If collapse is detected: increase α for load balance loss, or switch to expert-choice routing which guarantees balance. If token dropping is the issue: increase capacity_factor from 1.0 to 1.5.

**Q: How does expert parallelism differ from tensor parallelism?**  
A: Tensor parallelism splits a single weight matrix across GPUs — each GPU computes a partial result and all-reduces to get the full result. Every token uses every GPU. Expert parallelism assigns entire experts to different GPUs — each token only uses the GPU holding its assigned expert. EP requires all-to-all communication (each token goes to its expert's GPU, then returns), while TP requires all-reduce (sum across GPUs). EP is better when experts can fit on individual GPUs and all-to-all bandwidth is sufficient; TP is better when individual matrices are too large for one GPU and high-bandwidth NVLink is available.

## Flashcards

**Parameters?** #flashcard
determine model capacity and memory footprint

**Active FLOPs?** #flashcard
determine compute cost per token

**MoE decouples these?** #flashcard
large capacity at dense-model inference cost

**$f_i = \frac{\text{tokens dispatched to expert } i}{\text{total tokens}}$ (fraction of tokens)?** #flashcard
$f_i = \frac{\text{tokens dispatched to expert } i}{\text{total tokens}}$ (fraction of tokens)

**$P_i = \frac{1}{T}\sum_{x} G_i(x)$ (mean routing probability for expert i)?** #flashcard
$P_i = \frac{1}{T}\sum_{x} G_i(x)$ (mean routing probability for expert i)

**N = number of experts?** #flashcard
N = number of experts

**α = load balancing coefficient (typically 0.01–0.1)?** #flashcard
α = load balancing coefficient (typically 0.01–0.1)

**Shared experts (always active)?** #flashcard
handle common knowledge

**Routed experts (top-K selected)?** #flashcard
handle specialized knowledge

**capacity_factor ↑ → fewer dropped tokens, more memory?** #flashcard
capacity_factor ↑ → fewer dropped tokens, more memory

**capacity_factor ↓ → more dropped tokens (information loss), less memory?** #flashcard
capacity_factor ↓ → more dropped tokens (information loss), less memory

**Training?** #flashcard
drop tokens (ignore their gradients) at overloaded experts

**Inference?** #flashcard
typically use capacity_factor=2.0 to minimize drops, or use "no drop" mode

**Some experts activate on specific syntactic patterns (punctuation, numbers)?** #flashcard
Some experts activate on specific syntactic patterns (punctuation, numbers)

**Some on semantic domains (medical text, code, math)?** #flashcard
Some on semantic domains (medical text, code, math)

**Some on languages (English expert vs Chinese expert)?** #flashcard
Some on languages (English expert vs Chinese expert)
