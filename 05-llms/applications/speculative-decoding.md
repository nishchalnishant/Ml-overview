# Speculative Decoding — Faster LLM Inference

> How to get multiple tokens per forward pass without changing model quality. Covers standard speculative decoding, Medusa, Eagle, and production trade-offs.

---

## Table of Contents

1. [The Autoregressive Bottleneck](#1-the-autoregressive-bottleneck)
2. [Standard Speculative Decoding](#2-standard-speculative-decoding)
3. [Medusa — Multiple Draft Heads](#3-medusa--multiple-draft-heads)
4. [Eagle & Eagle-2 — Feature-Level Drafting](#4-eagle--eagle-2--feature-level-drafting)
5. [Self-Speculative Decoding](#5-self-speculative-decoding)
6. [Multi-Token Prediction (Built-in Speculation)](#6-multi-token-prediction-built-in-speculation)
7. [Production Trade-offs](#7-production-trade-offs)
8. [Interview Questions](#8-interview-questions)

---

## 1. The Autoregressive Bottleneck

LLMs generate one token at a time. Each token requires one full forward pass through all N transformer layers:

```
x = [t1, t2, ..., tn]
Forward pass → t_{n+1}   # 1 token per forward pass
Append t_{n+1} → [t1, ..., tn, t_{n+1}]
Forward pass → t_{n+2}   # another full pass
...
```

**Memory bandwidth is the bottleneck**, not compute:
- At batch size 1, GPU utilization is typically 1-5% of theoretical FLOPs
- The GPU is mostly reading model weights from HBM, not doing arithmetic
- A 70B model in bf16 = 140GB; reading 140GB per token at ~2TB/s HBM bandwidth = 70ms/token theoretical minimum

This is an **arithmetic intensity problem**: inference is memory-bound, not compute-bound. Strategies that generate multiple tokens per pass without increasing memory reads are pure wins.

---

## 2. Standard Speculative Decoding

### Core Idea (Chen et al., Leviathan et al., 2023)

Use a small, fast **draft model** to generate K candidate tokens. Then verify all K tokens with the large **target model** in a single forward pass. Under certain conditions, accepted tokens can be output at the cost of a single target model pass.

### The Algorithm

```
Target model:  M_q (the model you care about, e.g., LLaMA 70B)
Draft model:   M_p (small, fast, same vocabulary, e.g., LLaMA 7B)
Draft length:  K (typically 4-8)

For each decoding step:
  1. DRAFT: Run M_p autoregressively for K steps
     → draft tokens [x̃_1, x̃_2, ..., x̃_K] with probabilities [p_1, ..., p_K]

  2. VERIFY: Run M_q once on the full prefix + K draft tokens
     → target probabilities [q_1, ..., q_K, q_{K+1}] in a single forward pass

  3. ACCEPT/REJECT each draft token (left to right):
     For token i:
       acceptance probability = min(1, q_i(x̃_i) / p_i(x̃_i))
       - Sample u ~ Uniform[0,1]
       - If u ≤ acceptance_prob: ACCEPT x̃_i, continue to i+1
       - Else: REJECT. Sample a bonus token from adjusted distribution and STOP

  4. Output all accepted tokens + bonus token
```

### Acceptance Rate Mathematics

The expected number of tokens generated per target model call:

```
If acceptance rate per token = α:
  E[tokens per step] = K·α^K is NOT the right formula

Correct: E[tokens] = (1 - α^{K+1}) / (1 - α)

At α=0.8, K=4:  E[tokens] = (1 - 0.8^5) / (1 - 0.8) = (1 - 0.328) / 0.2 ≈ 3.36
At α=0.9, K=4:  E[tokens] = (1 - 0.9^5) / (1 - 0.9) = (1 - 0.590) / 0.1 ≈ 4.10
```

**Speedup ratio:** If the draft model is β times faster than the target model, the effective speedup is:

```
Speedup ≈ (E[tokens per step]) / (1 + K/β)
         ≈ (K·α) / (1 + K/β)    (approximation)
```

At α=0.8, K=4, β=10 (draft 10× faster): Speedup ≈ (4×0.8) / (1 + 0.4) ≈ 2.3×

### Correctness Guarantee

The algorithm produces **exactly the same distribution** as sampling from the target model alone. The rejection sampling step ensures: the output distribution is `q` regardless of how draft model `p` performs. This is not an approximation — speculative decoding is lossless.

```python
def speculative_decode_step(
    target_model, draft_model, input_ids, K=4, temperature=1.0
):
    # Step 1: Draft K tokens
    draft_ids = input_ids.clone()
    draft_probs = []
    for _ in range(K):
        with torch.no_grad():
            logits = draft_model(draft_ids).logits[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            draft_probs.append(probs[0, next_token.item()].item())
            draft_ids = torch.cat([draft_ids, next_token], dim=-1)
    
    # Step 2: Verify with target model (single forward pass)
    with torch.no_grad():
        target_logits = target_model(draft_ids).logits / temperature
        target_probs = torch.softmax(target_logits, dim=-1)
    
    # Step 3: Accept/reject
    accepted = input_ids.clone()
    for i in range(K):
        x_tilde = draft_ids[0, input_ids.shape[1] + i].item()
        q = target_probs[0, input_ids.shape[1] + i - 1, x_tilde].item()
        p = draft_probs[i]
        
        accept_prob = min(1.0, q / p)
        if torch.rand(1).item() <= accept_prob:
            accepted = torch.cat([accepted, draft_ids[:, input_ids.shape[1] + i : input_ids.shape[1] + i + 1]], dim=-1)
        else:
            # Resample from adjusted distribution
            adj_probs = torch.clamp(target_probs[0, input_ids.shape[1] + i - 1] - 
                                    torch.softmax(target_logits[0, input_ids.shape[1] + i - 1], dim=-1), min=0)
            adj_probs /= adj_probs.sum()
            bonus = torch.multinomial(adj_probs.unsqueeze(0), 1)
            accepted = torch.cat([accepted, bonus], dim=-1)
            break
    
    # If all K accepted, add one more from target
    if accepted.shape[1] == input_ids.shape[1] + K:
        bonus_probs = target_probs[0, -1]
        bonus = torch.multinomial(bonus_probs.unsqueeze(0), 1)
        accepted = torch.cat([accepted, bonus], dim=-1)
    
    return accepted
```

---

## 3. Medusa — Multiple Draft Heads

### Motivation

Standard speculative decoding requires maintaining two separate models — the draft model adds memory pressure and operational complexity. Medusa (Cai et al., 2024) attaches multiple **draft heads** directly to the target model.

### Architecture

Add K additional prediction heads to the final transformer layer, each predicting a different future token:

```
Transformer hidden state h_t (at token t)
                ↓
Head 0: W_0 · h_t → logits for token t+1  (standard LM head, unchanged)
Head 1: W_1 · h_t → logits for token t+2  (draft head 1)
Head 2: W_2 · h_t → logits for token t+3  (draft head 2)
...
Head K: W_K · h_t → logits for token t+K+1 (draft head K)
```

Each head is a simple linear layer on top of the frozen base model. Only the K heads are trained (not the base model).

### Tree Attention Verification

Instead of a linear draft (single candidate per step), Medusa generates a **tree** of candidates using top-s sampling at each head:

```
Tree structure (s=2 candidates per head, K=3 heads):
                    [Start]
                   /       \
               [cand_1_A] [cand_1_B]       ← Head 1 candidates
              /    \        /    \
           [2_A] [2_B]  [2_C] [2_D]        ← Head 2 candidates
          / \   / \    / \    / \
        [3...  3...  3...  3...]             ← Head 3 candidates

Total candidates: s^K = 2^3 = 8 paths
```

The target model verifies all paths in the tree simultaneously using a custom **tree attention mask** (each node can only attend to its ancestors):

```python
def build_tree_attention_mask(tree_structure):
    """Each tree node can attend to itself and all its ancestors."""
    N = len(tree_structure)
    mask = torch.zeros(N, N, dtype=torch.bool)
    for i, ancestors in enumerate(tree_structure):
        mask[i, ancestors] = True  # can attend to ancestors
        mask[i, i] = True          # can attend to self
    return mask
```

This is a single forward pass that validates multiple candidate paths — the longest valid prefix among all paths is selected.

### Medusa vs Standard Speculative Decoding

| Property | Standard SD | Medusa |
|----------|-------------|--------|
| Draft model memory | Separate model (e.g., 7B) | ~0 (just extra heads) |
| Acceptance rate | High (trained draft model) | Moderate (simple heads) |
| Training required | No | Yes (fine-tune heads) |
| Deployment complexity | Two models | One model |
| Speedup | 2-3× | 1.5-2.5× |

---

## 4. Eagle & Eagle-2 — Feature-Level Drafting

### Eagle (Li et al., 2024)

Eagle improves on Medusa by drafting at the **feature level** rather than the token level:

**Key insight:** The next-token prediction in the target model is most accurate if you can start from a hidden state that's close to the target model's actual hidden state at that position. Eagle trains a lightweight **draft model** that predicts the target model's hidden states:

```
Eagle architecture:
  Input: [token embeddings] + [previous target model hidden state]
  → Single transformer layer (the "eagle" model)
  → Predicted hidden state ĥ_{t+1}
  → LM head (shared with target model)
  → Draft token logits
```

By conditioning on the target model's hidden states (passed as additional context), Eagle's predictions are much better calibrated than Medusa's simple linear heads.

**Acceptance rate:** Eagle achieves ~0.85-0.90 acceptance rates (vs ~0.65-0.75 for Medusa) because it has access to the target model's actual intermediate representations.

### Eagle-2 — Dynamic Draft Tree

Eagle-2 improves Eagle by making the draft tree **dynamic** — the tree structure adapts based on confidence:

```python
def eagle2_draft(base_hidden, draft_model, lm_head, max_depth=5, threshold=0.5):
    """Dynamically grow the tree where confidence is high."""
    tree = [{"token": None, "hidden": base_hidden, "prob": 1.0}]
    
    for depth in range(max_depth):
        new_nodes = []
        for node in tree:
            if node["prob"] < threshold:
                continue  # prune low-confidence branches
            
            # Draft next token from current node
            logits = lm_head(draft_model(node["hidden"]))
            probs = torch.softmax(logits, dim=-1)
            
            # Expand top-k candidates where probability is significant
            top_probs, top_tokens = torch.topk(probs, k=3)
            for prob, token in zip(top_probs, top_tokens):
                if prob.item() * node["prob"] > threshold:
                    new_nodes.append({
                        "token": token,
                        "parent": node,
                        "prob": prob.item() * node["prob"],
                        "hidden": ...  # compute next hidden
                    })
        
        tree.extend(new_nodes)
    
    return tree
```

High-confidence branches get expanded (yielding longer accepted sequences); low-confidence branches get pruned (saving compute). Eagle-2 achieves 3-4× speedup vs 2-3× for Eagle on average.

---

## 5. Self-Speculative Decoding

### Layer Skipping

Instead of a separate draft model, use an **early exit** from the target model as the draft:

```
Standard: pass through all L layers
Self-speculative: 
  - Early exit at layer L/2 → draft token
  - Full pass through all L layers → verify
```

If the draft token matches the full model's prediction (acceptance rate ~0.7-0.8), you've effectively halved the compute for that token.

**Trade-off:** No extra parameters, no separate model. But acceptance rate is lower than a dedicated draft model trained on the same distribution. Works best when the model is "confident" (high-entropy tasks benefit less).

---

## 6. Multi-Token Prediction (Built-in Speculation)

### DeepSeek MTP

(Covered in `2025-frontier-models.md` under DeepSeek-V3.) Training auxiliary heads to predict tokens t+2, t+3 simultaneously enables built-in speculative decoding:

```
At inference:
  Forward pass → main head predicts t+1
              → aux head 1 predicts t+2 (speculative)
              → aux head 2 predicts t+3 (speculative)

If aux predictions match sequential decoding: multi-token advance
If rejected: fall back to standard one-token advance
```

This eliminates the need for any separate draft model while providing built-in speculation — the cost is baked into training.

---

## 7. Production Trade-offs

### When Speculative Decoding Helps

✅ **Small batch size (1-4):** Memory bandwidth is the bottleneck; speculative decoding multiplies tokens per memory read.

✅ **Long generation:** The amortized speedup is highest when generating many tokens (code generation, long documents).

✅ **High-quality draft model available:** A draft model from the same family (LLaMA 7B for LLaMA 70B target) has high acceptance rates.

### When Speculative Decoding Hurts

❌ **Large batch sizes:** At batch size 32+, the GPU is already compute-bound (high arithmetic intensity); speculative decoding adds coordination overhead without proportional speedup.

❌ **Very short responses:** The draft overhead isn't amortized over enough accepted tokens.

❌ **High-temperature sampling:** At temperature 1.5+, draft tokens are drawn from a different distribution than the target → lower acceptance rates.

### Acceptance Rate vs. Task

| Task | Typical Acceptance Rate | Speedup |
|------|------------------------|---------|
| Code completion | 0.85-0.92 | 2.5-4× |
| Factual QA | 0.75-0.85 | 2-3× |
| Creative writing | 0.60-0.75 | 1.5-2× |
| Math reasoning (extended CoT) | 0.55-0.70 | 1.3-1.8× |

Code generation benefits most because code has high local predictability (variable names repeat, boilerplate is formulaic).

### vLLM Speculative Decoding Config

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3.1-70B-Instruct",
    speculative_model="meta-llama/Llama-3.2-7B-Instruct",  # draft model
    num_speculative_tokens=5,   # K = draft length
    use_v2_block_manager=True,  # required for spec decoding in vLLM
)

sampling_params = SamplingParams(temperature=0.8, max_tokens=512)
outputs = llm.generate(["Write a Python function to sort a list"], sampling_params)
```

---

## 8. Interview Questions

**Q: What is the key correctness guarantee of speculative decoding?**

Speculative decoding produces the same output distribution as sampling directly from the target model, with zero approximation error. This is guaranteed by the rejection sampling step: when a draft token is rejected, we resample from the adjusted distribution `max(0, q - p) / Z`, which combined with the already-accepted tokens preserves the target distribution exactly. You can verify this mathematically: the marginal probability of any output sequence under speculative decoding equals the probability under direct target model sampling.

---

**Q: Why does speculative decoding help more at batch size 1 than batch size 32?**

At batch size 1, LLM inference is **memory-bandwidth bound** — the GPU spends most of its time reading model weights from HBM rather than doing arithmetic. Generating K draft tokens + 1 verification pass reads weights once from HBM (for the target model) and K extra times from HBM for the draft model, but outputs K+1 tokens instead of 1. At large batch sizes, the GPU is **compute-bound** (weights are read once and reused across the batch). Speculative decoding adds overhead (draft model forward passes, verification coordination) that doesn't improve the already-high arithmetic intensity, so it provides minimal benefit or even slight regression.

---

**Q: What's the difference between Medusa and Eagle architecturally?**

Medusa attaches K independent linear heads directly to the final hidden state of the target model, each predicting a different future token position. The heads see only the current hidden state — no recurrent context. Eagle is a separate single-layer transformer that takes both the current token embedding AND the target model's hidden state as input. By conditioning on the target model's actual hidden representation, Eagle can make much better-calibrated predictions about what the target model would output next. The trade-off: Eagle requires a full autoregressive forward pass of the eagle model, while Medusa is just K linear layer forward passes.

---

**Q: At what acceptance rate does speculative decoding break even with standard decoding?**

The break-even depends on the relative cost of draft vs. target. If the draft model is β times cheaper per token, speculative decoding with K draft tokens breaks even when:

```
E[tokens per step] / (1 + K/β) ≥ 1

Substituting E[tokens] ≈ K·α (at high acceptance rates):
α ≥ (1 + K/β) / K = 1/K + 1/β
```

At K=4, β=10: break-even at α ≥ 1/4 + 1/10 = 0.35. In practice, any reasonable draft model achieves α > 0.5, so speculative decoding almost always helps at small batch sizes when a good draft model is available.

---

*Last updated: May 2026 | Coverage: Standard SD, Medusa, Eagle, MTP | Focus: algorithm + production*
