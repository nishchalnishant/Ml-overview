---
module: LLMs
topic: Kv Cache And Mqa Gqa
subtopic: ""
status: unread
tags: [llms, ml, kv-cache-and-mqa-gqa]
---
# KV Cache, MQA, GQA, and Memory-Efficient Attention

How Transformer inference memory scales and the architectures used to reduce it.

> For GQA architecture basics and a PyTorch implementation in the context of the full transformer block, see [architecture-deep-dive.md § 3](03-architecture-deep-dive.md). This file covers the memory analysis, MLA (DeepSeek), prefix caching, and KV quantization in depth.

---

## 1. Standard Multi-Head Attention (MHA)

For each layer, attention computes:
$$Q = X W_Q, \quad K = X W_K, \quad V = X W_V$$

Each of Q, K, V has shape [seq, n_heads, d_k] where d_k = d_model / n_heads.

**KV cache at generation step t:**
- Cache all K, V from positions 1..t
- Reuse instead of recomputing

**Memory per token, per layer (single K or V matrix):**
$$\text{bytes} = n_{heads} \times d_k \times \text{dtype\_bytes}$$

**Full KV cache:**
$$\text{KV bytes} = 2 \times n_{layers} \times n_{heads} \times d_k \times S_{context} \times \text{dtype\_bytes}$$

---

## 2. Multi-Query Attention (MQA)

**Key idea (Shazeer 2019):** All query heads share a single K and V head.

```
MHA:  Q₁ K₁ V₁
      Q₂ K₂ V₂   ← H separate K,V pairs
      ...
      Qₕ Kₕ Vₕ

MQA:  Q₁ K  V
      Q₂ K  V   ← single shared K,V for all queries
      ...
      Qₕ K  V
```

**Memory savings:**
$$\frac{\text{MQA KV}}{\text{MHA KV}} = \frac{1}{n_{heads}}$$

For H=32 heads: MQA uses **32× less** KV cache memory.

**Quality cost:** Slight degradation, especially on tasks requiring diverse attention patterns. PaLM (2022) first major model to use MQA.

```python
class MultiQueryAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None):
        super().__init__()
        d_k = d_k or d_model // n_heads
        self.W_Q = nn.Linear(d_model, n_heads * d_k)  # H query heads
        self.W_K = nn.Linear(d_model, d_k)            # 1 key head (shared)
        self.W_V = nn.Linear(d_model, d_k)            # 1 value head (shared)
        self.W_O = nn.Linear(n_heads * d_k, d_model)
        self.n_heads = n_heads
        self.d_k = d_k
    
    def forward(self, x, kv_cache=None):
        B, S, _ = x.shape
        Q = self.W_Q(x).view(B, S, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(x).unsqueeze(2)  # [B, S, 1, d_k] → broadcast to all heads
        V = self.W_V(x).unsqueeze(2)
        
        if kv_cache is not None:
            K = torch.cat([kv_cache['k'], K], dim=1)
            V = torch.cat([kv_cache['v'], V], dim=1)
        
        # K,V broadcast across n_heads via expand (no copy)
        K_expanded = K.expand(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        V_expanded = V.expand(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K_expanded.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V_expanded)
        out = out.transpose(1, 2).contiguous().view(B, S, -1)
        return self.W_O(out), {'k': K, 'v': V}
```

---

## 3. Grouped-Query Attention (GQA)

**Key idea (Ainslie et al. 2023):** Intermediate between MHA and MQA. Group query heads into G groups, each group shares one K,V head.

```
GQA (G=4, H=32):
  Group 1: Q₁ Q₂ Q₃ Q₄ Q₅ Q₆ Q₇ Q₈  → share K₁ V₁
  Group 2: Q₉ ... Q₁₆                  → share K₂ V₂
  Group 3: Q₁₇ ... Q₂₄                 → share K₃ V₃
  Group 4: Q₂₅ ... Q₃₂                 → share K₄ V₄
```

**Number of KV heads:** n_kv_heads = G = H / (heads_per_group)

**Memory savings:**
$$\frac{\text{GQA KV}}{\text{MHA KV}} = \frac{G}{H} = \frac{n_{kv\_heads}}{n_{heads}}$$

For Llama-3 8B: H=32 query heads, G=8 KV heads → **4× memory reduction** vs MHA.

**Memory formula (GQA):**
$$\text{KV bytes} = 2 \times n_{layers} \times n_{kv\_heads} \times d_k \times S_{context} \times \text{dtype\_bytes}$$

**Examples:**
| Model | n_heads | n_kv_heads | KV reduction |
|---|---|---|---|
| Llama-2 7B | 32 | 32 (MHA) | 1× |
| Llama-3 8B | 32 | 8 (GQA) | 4× |
| Llama-3 70B | 64 | 8 (GQA) | 8× |
| Mistral 7B | 32 | 8 (GQA) | 4× |
| Falcon-40B | 64 | 8 (MQA-like) | 8× |

```python
class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads):
        super().__init__()
        assert n_heads % n_kv_heads == 0
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads  # heads per group
        d_k = d_model // n_heads
        
        self.W_Q = nn.Linear(d_model, n_heads * d_k)
        self.W_K = nn.Linear(d_model, n_kv_heads * d_k)  # fewer heads
        self.W_V = nn.Linear(d_model, n_kv_heads * d_k)
        self.W_O = nn.Linear(n_heads * d_k, d_model)
        self.d_k = d_k
    
    def repeat_kv(self, x):
        """Expand KV heads to match Q heads."""
        # x: [B, S, n_kv_heads, d_k] → [B, S, n_heads, d_k]
        B, S, n_kv, d_k = x.shape
        return x.unsqueeze(3).expand(B, S, n_kv, self.n_rep, d_k).reshape(B, S, -1, d_k)
    
    def forward(self, x):
        B, S, _ = x.shape
        Q = self.W_Q(x).view(B, S, self.n_heads, self.d_k)
        K = self.W_K(x).view(B, S, self.n_kv_heads, self.d_k)
        V = self.W_V(x).view(B, S, self.n_kv_heads, self.d_k)
        
        # Repeat K,V to match Q head count
        K = self.repeat_kv(K)
        V = self.repeat_kv(V)
        
        # Standard attention
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = F.softmax(scores, dim=-1)
        return self.W_O(torch.matmul(attn, V).transpose(1, 2).reshape(B, S, -1))
```

---

## 4. Prefix Caching

**Use case:** Multiple requests share the same prefix (system prompt, few-shot examples, RAG chunks).

**Optimization:** Compute and cache the KV for the shared prefix once, reuse for all requests with that prefix.

```
Request 1: [system_prompt | user_query_1]
Request 2: [system_prompt | user_query_2]
Request 3: [system_prompt | user_query_3]
                ↑
         Cached KV for system_prompt
         TTFT reduces by (prefix_len / total_len) fraction
```

**Savings:**
- TTFT reduction: prefix tokens skip prefill computation
- KV memory: prefix KV shared across N requests (effective 1/N memory overhead)

**Implementation:** Use hash of prefix tokens as cache key. vLLM supports automatic prefix caching.

```python
class PrefixCache:
    def __init__(self, max_entries=1000):
        self.cache = {}  # token_hash → KV tensors
    
    def get(self, tokens):
        # Find longest cached prefix
        for length in range(len(tokens), 0, -1):
            key = hash(tuple(tokens[:length]))
            if key in self.cache:
                return self.cache[key], length
        return None, 0
    
    def put(self, tokens, kv_tensors):
        key = hash(tuple(tokens))
        self.cache[key] = kv_tensors
        # Evict LRU if over capacity
```

---

## 5. DeepSeek MLA (Multi-Head Latent Attention)

**Problem:** Even GQA requires n_kv_heads × n_layers × d_k × context KV memory. At long contexts, this dominates GPU memory.

**MLA insight (DeepSeek-V2, 2024):** Compress K and V to a shared low-rank latent vector.

**Architecture:**
$$c_{KV} = W_{DKV} x \quad \in \mathbb{R}^{d_{c}}$$
$$K = W_{UK} c_{KV}, \quad V = W_{UV} c_{KV}$$

where d_c << n_kv_heads × d_k is the compressed KV dimension.

**Memory comparison:**

| Method | KV memory per token |
|---|---|
| MHA (H=128, d_k=128) | 2 × 128 × 128 × 2B = 65,536 bytes |
| GQA (n_kv=8) | 2 × 8 × 128 × 2B = 4,096 bytes |
| MLA (d_c=512) | 512 × 2B = 1,024 bytes |

**MLA compression ratio vs MHA:**
$$\frac{d_c}{2 \times n_{kv\_heads} \times d_k} = \frac{512}{2 \times 128 \times 128} = \frac{1}{64}$$

DeepSeek claims **13.5× reduction** vs standard MHA in practice (accounting for additional RoPE key).

**The key trick:** Store only c_KV (the compressed latent) in the KV cache. At attention time, reconstruct K and V on the fly via W_UK and W_UV. Trade: small compute overhead vs large memory savings.

```python
class MultiHeadLatentAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_c, d_kv_rope):
        """
        d_c: compressed KV dimension (the key hyperparameter)
        d_kv_rope: decoupled RoPE dimension for keys
        """
        super().__init__()
        d_k = d_model // n_heads
        
        # KV compression
        self.W_DKV = nn.Linear(d_model, d_c + d_kv_rope)  # down-projection
        self.W_UK = nn.Linear(d_c, n_heads * d_k)           # K up-projection
        self.W_UV = nn.Linear(d_c, n_heads * d_k)           # V up-projection
        
        # Q compression (optional, for further savings)
        self.W_DQ = nn.Linear(d_model, d_c)
        self.W_UQ = nn.Linear(d_c, n_heads * d_k)
        
        # Standard Q projection for RoPE
        self.W_QR = nn.Linear(d_model, n_heads * d_kv_rope)
        
    def forward(self, x, kv_cache=None):
        # Compress KV (only this is cached)
        kv_compressed = self.W_DKV(x)
        c_KV, k_rope = kv_compressed.split([self.d_c, self.d_kv_rope], dim=-1)
        
        if kv_cache is not None:
            # Append compressed representation (much smaller than full KV)
            c_KV = torch.cat([kv_cache['c_KV'], c_KV], dim=1)
            k_rope = torch.cat([kv_cache['k_rope'], k_rope], dim=1)
        
        # Reconstruct K, V from compressed latent at attention time
        K = self.W_UK(c_KV)  # not cached, computed on the fly
        V = self.W_UV(c_KV)
        
        new_cache = {'c_KV': c_KV, 'k_rope': k_rope}
        # ... compute attention ...
        return output, new_cache
```

---

## 6. KV Cache Quantization

**Observation:** KV cache is stored in fp16 (2 bytes/element) by default, but can be quantized:

| Precision | Bytes/element | Quality loss |
|---|---|---|
| fp16 | 2 | baseline |
| int8 | 1 | negligible (<0.1% on benchmarks) |
| int4 | 0.5 | small (~0.5–1%) |
| int4 with outlier channels in fp16 | ~0.6 | minimal |

**Per-request savings:**
- Llama-3 8B at 8K context, fp16 → int8: 536MB → 268MB per request
- Doubles concurrent batch size on same hardware

---

## 7. Memory Summary: Llama-3 8B Serving

```
Component                    Memory (fp16)
─────────────────────────────────────────
Model weights                  16 GB
KV cache (batch=50, ctx=8K)    50 × 536MB = 26.8 GB
Activation buffers             ~2 GB
──────────────────────────────────────────
Total                          ~45 GB

On A100 80GB: leaves ~35GB headroom
→ can increase to batch≈115 concurrent requests

With int8 KV cache:
KV cache (batch=100, ctx=8K)   100 × 268MB = 26.8 GB (same, but 2× requests)
→ batch=200 concurrent requests on 80GB A100
```

---

## Canonical Interview Q&As

**Q: Derive the memory savings of GQA over MHA for Llama-3 70B at 4K context.**  
A: MHA KV bytes = 2 × L × H × d_k × S × dtype_bytes = 2 × 80 × 64 × 128 × 4096 × 2 = 10.7 GB per request. GQA (n_kv=8): 2 × 80 × 8 × 128 × 4096 × 2 = 1.34 GB per request. Savings = 8×. In practice, this is the difference between fitting ~6 requests vs ~50 requests in an 80GB A100 (after subtracting 35GB for model weights). GQA is what makes 70B serving practical with reasonable batch sizes.

**Q: What is the quality trade-off between MHA, GQA, and MQA?**  
A: MHA gives the best attention expressivity — each head can specialize to different token relationships. MQA (1 KV head) is the most extreme compression; it degrades on tasks requiring diverse attention patterns (multi-hop reasoning, structured output). GQA (8 KV heads) recovers most of the quality gap: empirically within 1–2% of MHA on standard benchmarks, while providing 8–32× KV memory savings. The right trade-off is model-specific: Llama-3 uses n_kv=8 for 70B (8× savings, minimal quality loss). For 8B, 4× savings is still significant. MQA is now rarely used in new models.

**Q: How does DeepSeek MLA achieve 13.5× KV compression vs MHA?**  
A: MLA stores only the compressed latent c_KV ∈ ℝ^{d_c} per token, not full K and V matrices. For DeepSeek-V2: d_model=5120, H=128, d_k=128, d_c=512. Standard MHA KV per token = 2 × 128 × 128 = 32,768 floats. MLA KV per token = 512 floats. Ratio = 64×. The "13.5×" figure accounts for the additional decoupled RoPE key component and implementation overhead. At attention time, K and V are reconstructed from c_KV via learned up-projection matrices — adds FLOPs but dramatically reduces KV cache size.

**Q: When would prefix caching not help, and what are its failure modes?**  
A: Prefix caching only helps when multiple requests share an identical token prefix. Failure modes: (1) slight prefix variation (different system prompt version → cache miss) — requires exact token match; (2) high cardinality — if each user has a personalized system prompt, cache hit rate is low; (3) memory pressure — the prefix KV must stay cached long enough to be reused; under heavy load, LRU eviction may evict prefixes before they're hit. Mitigation: pin high-value prefixes (shared system prompts for all users), use session affinity routing (send same user's requests to same server to hit warm cache).

## Flashcards

**What does the KV cache store, and why does it avoid recomputation?** #flashcard
It caches K and V for every position 1..t computed so far in generation, so each new decoding step reuses them instead of recomputing attention over the whole prefix.

**How does prefix caching reduce both latency and memory for shared prompts?** #flashcard
TTFT drops because the shared prefix (e.g. system prompt) skips prefill computation entirely. KV memory drops because the prefix's KV is cached once and shared across N requests, giving an effective 1/N memory overhead per request.

**What's the effect of quantizing the KV cache from fp16 to int8?** #flashcard
Roughly halves KV cache memory per request (e.g. Llama-3 8B at 8K context: 536MB → 268MB), which doubles the concurrent batch size servable on the same hardware, with negligible quality loss.
