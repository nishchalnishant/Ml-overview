---
module: Systems and Scale
topic: System Design
subtopic: Llm Inference Ops
status: unread
tags: [productionml, ml, system-design-llm-inference-op]
---
# LLM Inference Operations

Production serving of LLMs at scale. Covers TTFT/TPOT math, memory management, batching strategies, and capacity planning.

---

## 1. Key Metrics

### TTFT — Time to First Token
Time from request submission to first output token streamed to user.

$$\text{TTFT} = t_{queue} + t_{prefill}$$

- **t_queue:** waiting for a server to accept the request (batch scheduling delay)
- **t_prefill:** processing the entire input prompt (compute-bound)

### TPOT — Time Per Output Token
Average time to generate each subsequent token after the first.

$$\text{TPOT} = t_{queue\_decode} + t_{decode\_per\_token}$$

- **t_decode_per_token:** one forward pass of the decoder (memory-bandwidth-bound)

### E2E Latency
$$\text{E2E Latency} = \text{TTFT} + (L_{output} - 1) \times \text{TPOT}$$

For a 1000-token output with TTFT=500ms, TPOT=50ms:
- E2E = 500 + 999 × 50 = 50.45 seconds

### Throughput
$$\text{Throughput} = \frac{\text{tokens\_per\_second}}{\text{cost\_per\_GPU}}$$

Reported as tokens/sec/GPU. Batching is the primary lever.

---

## 2. Prefill vs Decode: Compute Analysis

### Prefill (compute-bound)

Processing S input tokens simultaneously — parallelizable across the sequence.

**FLOPs:** For each attention layer with n_heads, d_model:
$$\text{FLOPs}_{attn} = 4 \cdot S^2 \cdot d_{model} + \text{FFN terms}$$

Total per layer ≈ 12 × S × d²_model (attention + FFN for standard 4× expansion).

**Bottleneck:** Matrix multiplications with full sequence. Compute-bound once S > ~100 tokens.

```
Arithmetic intensity (prefill, seq S):
  FLOPs = 2 × N × S   (N = total params)
  Bytes = N × dtype_bytes  (load weights)
  Intensity = 2S ops/byte

  For S=2048 tokens, A100 (arithmetic intensity threshold ~500):
  2×2048 = 4096 >> 500 → compute-bound ✓
```

### Decode (memory-bandwidth-bound)

Generating one token at a time — cannot parallelize across output sequence (autoregressive).

**FLOPs per token:** 2 × N (one forward pass).

**Bytes loaded:** N × dtype_bytes (load all model weights for every single token).

```
Arithmetic intensity (decode, batch=1):
  FLOPs = 2N
  Bytes = N × 2  (fp16)
  Intensity = 1 op/byte << A100 threshold (500)
  → Memory bandwidth bound ✓

At batch=B:
  FLOPs = 2NB
  Bytes = N × 2  (weights shared across batch)
  Intensity = B ops/byte
  → Compute-bound when B > ~500 (A100)
```

**Implication:** Decode throughput scales near-linearly with batch size up to the compute threshold. Pack as many requests as possible.

---

## 3. KV Cache

### Why KV Cache Exists

In autoregressive decoding, at step t we need K and V from all previous positions 1..t. Recomputing from scratch = O(t²) total work. Cache them = O(t) incremental work.

### Memory Formula

$$\text{KV cache bytes} = 2 \times L \times H \times d_k \times S_{context} \times \text{dtype\_bytes}$$

Where:
- 2 = K and V
- L = num layers
- H = num KV heads
- d_k = head dimension
- S_context = max context tokens

**Example: Llama-3 8B (fp16)**
- L=32, H=8 (GQA), d_k=128, context=8192
- KV = 2 × 32 × 8 × 128 × 8192 × 2 = **536 MB per request**

**At batch_size=100:**
- KV = 53.6 GB — exceeds A100 80GB even before model weights (~16GB)

**Llama-3 70B (fp16)**
- L=80, H=8 (GQA), d_k=128, context=4096
- KV = 2 × 80 × 8 × 128 × 4096 × 2 = **671 MB per request**

### KV Cache Capacity Planning

```python
def kv_cache_memory_gb(num_layers, num_kv_heads, head_dim, 
                        context_len, batch_size, dtype_bytes=2):
    """Compute KV cache memory in GB."""
    bytes_per_token = 2 * num_layers * num_kv_heads * head_dim * dtype_bytes
    total_bytes = bytes_per_token * context_len * batch_size
    return total_bytes / 1e9

# Available memory for KV cache:
def max_batch_size(gpu_memory_gb, model_memory_gb, **kv_kwargs):
    available = (gpu_memory_gb - model_memory_gb) * 1e9
    bytes_per_seq = kv_cache_memory_gb(batch_size=1, **kv_kwargs) * 1e9
    return int(available / bytes_per_seq)

# Llama-3 8B on 80GB A100:
# Model weights: ~16GB (fp16)
# Available: 64GB
# Per-seq KV at 8K context: 536MB
# Max batch: 64GB / 536MB ≈ 119 concurrent requests
```

---

## 4. PagedAttention

**Problem:** Traditional KV cache pre-allocates a contiguous block of memory for max_context_len per request. With variable sequence lengths, most of this is wasted — fragmentation wastes 40–80% of KV cache.

**Solution (vLLM, Kwon et al. 2023):** Virtual memory for KV cache, inspired by OS paging.

```
Physical KV memory:
┌──────┬──────┬──────┬──────┬──────┬──────┐
│ blk0 │ blk1 │ blk2 │ blk3 │ blk4 │ blk5 │  ← fixed-size physical blocks (16 tokens each)
└──────┴──────┴──────┴──────┴──────┴──────┘

Request A (300 tokens → 19 blocks):
  block table: [blk0, blk3, blk5, ...]  ← logical → physical mapping

Request B (50 tokens → 4 blocks):
  block table: [blk1, blk2, blk4, ...]  ← no fragmentation between requests
```

**Key operations:**
- **Allocate:** assign physical blocks to new request (on demand, not upfront)
- **Free:** return blocks when request completes
- **Copy-on-write:** for beam search, parent and child share blocks until divergence

**Result:** 2–4× higher throughput vs HuggingFace transformers (less memory waste → larger effective batch size).

```python
# PagedAttention pseudocode
class BlockManager:
    def __init__(self, total_blocks, block_size=16):
        self.free_blocks = list(range(total_blocks))
        self.block_tables = {}  # request_id → [physical_block_ids]
    
    def allocate(self, request_id, num_tokens):
        n_blocks = math.ceil(num_tokens / self.block_size)
        blocks = [self.free_blocks.pop() for _ in range(n_blocks)]
        self.block_tables[request_id] = blocks
    
    def can_allocate(self, num_tokens):
        return len(self.free_blocks) >= math.ceil(num_tokens / self.block_size)
    
    def free(self, request_id):
        self.free_blocks.extend(self.block_tables.pop(request_id))
```

---

## 5. Continuous Batching

**Static batching problem:** Old servers wait until a batch is full before processing. If requests have different lengths, fast requests block on slow ones:

```
Batch: [req_A (10 tokens), req_B (500 tokens), req_C (20 tokens)]
       ← all three wait for req_B to finish → GPU is idle for A and C
```

**Continuous batching (iteration-level scheduling, Yu et al. 2022):**

```
Step 1: [req_A, req_B, req_C]  → generate 1 token each
Step 2: req_A finishes → immediately insert req_D
        [req_D, req_B, req_C]  → generate 1 token each
Step 3: req_C finishes → insert req_E
        [req_D, req_B, req_E]  → ...
```

Batch is re-formed at every token generation step. New requests fill slots vacated by completed ones.

**Result:** GPU utilization increases from ~30% (static) to >80% (continuous batching).

```python
# Simplified continuous batching scheduler pseudocode
class ContinuousBatchScheduler:
    def __init__(self, max_batch_size, block_manager):
        self.running = []   # actively generating
        self.waiting = []   # queued requests
        self.block_manager = block_manager
    
    def schedule_step(self):
        # Preempt if memory pressure
        while not self.block_manager.can_allocate(1) and self.running:
            req = self.running.pop()  # preempt lowest priority
            self.waiting.insert(0, req)
        
        # Admit waiting requests into running
        while (len(self.running) < self.max_batch_size and 
               self.waiting and
               self.block_manager.can_allocate(1)):
            req = self.waiting.pop(0)
            self.block_manager.allocate(req.id, req.current_length)
            self.running.append(req)
        
        return self.running  # batch for this step
```

---

## 6. Chunked Prefill

**Problem:** Long prompts (8K–128K tokens) cause TTFT spikes and block decode operations.

**Solution:** Split the prefill phase into chunks (e.g., 512 tokens each), interleave with decode:

```
Without chunked prefill:
  [8K token prefill: 800ms] → [decode start: 200ms/token]
  TTFT = 800ms

With chunked prefill (chunk=512):
  [512 tokens prefill] → [decode step 1] → [512 tokens prefill] → [decode step 2] → ...
  TTFT = 50ms (first chunk done), TPOT increases slightly (interleaved)
```

**Trade-off:** Reduces TTFT, slightly increases TPOT due to context switching. Configure chunk size based on TTFT SLA.

---

## 7. Speculative Decoding

**Problem:** Decode is memory-bound with batch=1. We can't speed up a single token step beyond memory bandwidth.

**Solution (Chen et al. 2023):** Use a small draft model to speculatively generate K tokens, then verify with large model in one parallel forward pass.

**Algorithm:**
```
Draft model (e.g., 68M) generates tokens [t₁, t₂, ..., tₖ]
Target model (e.g., 7B) verifies all K tokens in ONE forward pass (parallel!)
Accept tᵢ with probability min(1, p_target(tᵢ) / p_draft(tᵢ))
On rejection: resample from corrected distribution, discard subsequent tokens
```

**Expected accepted tokens per step:**
$$\mathbb{E}[\text{accepted}] = \frac{1 - \alpha^{K+1}}{1 - \alpha}$$

where α = average acceptance rate per token.

For α=0.8, K=5: E[accepted] = (1 - 0.8⁶)/(1 - 0.8) = 0.738/0.2 ≈ 3.7 tokens per target-model call.

```python
def speculative_decode(draft_model, target_model, prompt, K=5):
    """Simplified speculative decoding."""
    generated = []
    
    while not done:
        # Draft: generate K candidate tokens autoregressively
        draft_tokens = []
        draft_probs = []
        context = prompt + generated
        for _ in range(K):
            p_draft = draft_model(context)
            t = sample(p_draft)
            draft_tokens.append(t)
            draft_probs.append(p_draft[t])
            context = context + [t]
        
        # Target: verify all K tokens in ONE forward pass
        all_context = prompt + generated + draft_tokens
        target_logits = target_model(all_context)  # parallel!
        target_probs = softmax(target_logits[len(generated):len(generated)+K])
        
        # Accept/reject
        accepted = []
        for i, (t, p_d, p_t) in enumerate(zip(draft_tokens, draft_probs, target_probs)):
            acceptance_rate = min(1.0, p_t[t] / p_d)
            if random() < acceptance_rate:
                accepted.append(t)
            else:
                # Resample from corrected distribution
                t_new = sample(max(0, p_t - p_d))
                accepted.append(t_new)
                break  # discard rest
        
        generated.extend(accepted)
    
    return generated
```

**Speedup (Eagle-2 style, 2024):** 3–4× speedup using a draft model that conditions on target model's hidden states (better alignment → higher acceptance rate).

---

## 8. Model Routing and Inference Tiers

Not every request needs the biggest model. Route by complexity:

```
Request → Router → Small model (7B)  ← simple queries, low cost
               └─► Medium model (70B) ← moderate complexity
               └─► Large model (405B) ← expert-level, high cost
```

**Routing signals:**
- Input length (longer → harder)
- User tier (premium → larger model)
- Task type (code → code-specialized, multimodal → vision model)
- Confidence from small model (if >0.95 confidence, don't escalate)

```python
class InferenceRouter:
    def route(self, request):
        # LLM-based routing: cheap classifier to decide
        complexity = self.classify_complexity(request.prompt)
        
        if complexity == "simple" and len(request.prompt) < 500:
            return "7b-fast"
        elif request.user_tier == "premium" or complexity == "complex":
            return "70b-quality"
        else:
            return "13b-balanced"
```

---

## 9. Capacity Planning

**Given:** Serve 10K QPS with average 1K input + 500 output tokens, P99 TPOT < 100ms.

**Step 1: Throughput requirement**
- Tokens/sec = 10K × (1K + 500) = 15M tokens/sec total throughput
- Effective decode tokens/sec = 10K × 500 = 5M tokens/sec

**Step 2: Per-GPU throughput**
- A100 80GB, Llama-3 70B (4-bit quantized, ~35GB model)
- At batch_size=32, fp8: ~3000 decode tokens/sec per GPU (empirical)

**Step 3: GPU count**
- GPUs needed = 5M / 3000 ≈ 1667 GPUs
- With 8× A100 nodes: ~209 nodes

**Step 4: KV cache check**
- Each request holds 1.5K tokens × 671MB/8K × (1.5K/8K) = ~126MB
- At 10K concurrent requests: 1.26 TB KV cache → distributed across nodes

```python
def capacity_plan(qps, avg_input_tokens, avg_output_tokens, 
                  model_name, gpus_per_node=8):
    # Empirical decode throughput (tokens/sec/GPU) from benchmarks
    throughput_per_gpu = {
        "llama-3-8b-fp16": 5000,
        "llama-3-70b-fp8": 3000,
        "llama-3-405b-fp8": 800
    }
    
    decode_tps_required = qps * avg_output_tokens
    gpus = math.ceil(decode_tps_required / throughput_per_gpu[model_name])
    nodes = math.ceil(gpus / gpus_per_node)
    
    print(f"Decode TPS required: {decode_tps_required:,}")
    print(f"GPUs needed: {gpus:,}")
    print(f"Nodes needed: {nodes:,}")
    return nodes
```

---

## Trade-offs Summary

| Decision | Trade-off |
|---|---|
| Batch size | Larger → better throughput, worse latency |
| Chunked prefill chunk size | Smaller → better TTFT, slightly worse TPOT |
| Speculative decoding K | Larger K → more speedup at low load, overhead at high load |
| Quantization (fp8 vs fp16) | fp8 → 2× throughput, <0.5% quality loss |
| PagedAttention block size | Larger blocks → less overhead, more wasted memory on short seqs |
| Draft model size (speculative) | Larger → higher acceptance, slower draft step |

---

## Interview Angles

### Q: Why is LLM decode memory-bandwidth-bound but prefill compute-bound? [Hard]  
A: During decode (batch=1), we generate one token per step. The FLOPs required = 2N (one forward pass), but we must load all N model parameters from HBM. Arithmetic intensity = 1 op/byte, far below A100's threshold (~500). At batch size B, intensity = B ops/byte — becomes compute-bound near B=500. Prefill processes S tokens simultaneously; matrix multiply is O(N×S) FLOPs but still O(N) weight load, so intensity = S. Once S > a few hundred, prefill is compute-bound and GPU utilization is high.

**Cross-questions to expect:**

- *You say batching decode to B~500 makes it compute-bound. Why don't real servers just crank batch size until decode is compute-bound and call it solved?* -> Because KV-cache memory grows linearly with batch size and context length, so long before B=500 you run out of HBM to hold all those sequences' caches -- you hit the memory-*capacity* wall before you reach the compute-bound regime. Decode is bandwidth-bound in practice not because you can't imagine a large batch, but because KV capacity caps the batch you can actually assemble. The two limits fight each other.
- *If decode is bandwidth-bound at batch=1, does quantizing weights to int4 give a clean 4x decode speedup?* -> No -- it helps because you load fewer weight bytes, but decode also loads the *KV cache* from HBM, and quantizing weights doesn't shrink that. As context grows, KV traffic becomes a large share of the bytes moved, so the weight-quantization speedup is capped by the un-quantized KV reads. This is why long-context decode benefits from KV-cache quantization specifically, not just weight quantization.

**Trap:** treating "arithmetic intensity" as a fixed property of the model. It's a property of the *workload* -- batch size, context length, and which tensors dominate the byte traffic all move the intensity, so the same model is bandwidth-bound in one serving regime and compute-bound in another.
**Q: How does PagedAttention improve throughput?**  
A: Traditional KV cache pre-allocates contiguous memory for max_context_len per request. With variable-length requests, gaps between requests cause fragmentation — up to 70% of KV memory is wasted on internal fragmentation and padding. PagedAttention uses fixed-size physical blocks (16 tokens) and a logical-to-physical block table per request, like OS virtual memory. Blocks are allocated on demand and freed immediately when requests complete. This eliminates fragmentation, enabling 2–4× more concurrent requests → 2–4× higher throughput for the same hardware.

### Q: You need to reduce TTFT for a chatbot handling 10K-token system prompts. What do you do? [Medium]  
A: (1) Chunked prefill: process the 10K prompt in 512-token chunks, interleaved with decode steps — first token arrives after 512 tokens instead of 10K; (2) Prefix caching: if the system prompt is shared across requests (e.g., same persona), cache its KV computation — TTFT for subsequent requests with the same prefix is near-zero; (3) Request priority queue: deprioritize long prefills during high-load periods to not block waiting decode requests; (4) Parallel prefill: for very long prompts, split across multiple GPUs (tensor parallel for prefill phase only).

**Cross-questions to expect:**

- *Chunked prefill improves TTFT by interleaving prefill with decode. What are you taking away from, and who notices?* -> You're stealing compute from the in-flight *decode* requests to make room for prefill chunks, so their inter-token latency (TPOT) rises -- you improve the newcomer's time-to-first-token at the cost of every currently-streaming user's smoothness. It's a latency *reallocation*, not a free win, and under load the tradeoff between TTFT and TPOT has to be tuned deliberately, not assumed favorable.
- *Prefix caching makes shared system prompts near-free. When does it silently stop helping or become a liability?* -> When prompts aren't actually identical -- a single differing token near the start invalidates the shared prefix, so per-user personalization injected early kills the cache hit. And the cached KV blocks consume the same scarce HBM the KV cache needs for live sequences, so an aggressive prefix cache can *reduce* the concurrency you can serve. It's a memory-for-latency trade with a capacity cost, not pure upside.

**Trap:** optimizing TTFT in isolation. TTFT and TPOT (inter-token latency) trade against each other through the shared compute budget -- a chatbot that first-tokens instantly but then streams slowly can feel worse than one with a slightly longer initial pause. The SLA has to name both.
**Q: What is continuous batching and how does it compare to static batching?**  
A: Static batching: form a batch, wait for all requests to complete, form next batch. If batch has a 500-token request and a 10-token request, GPU sits idle after the short request finishes. Continuous batching re-forms the batch at every token generation step: when a request completes, the slot is immediately filled with a queued request. This keeps GPU utilization continuously high (>80% vs ~30% for static). Implemented in vLLM, TGI — essential for production serving with heterogeneous request lengths.

## Flashcards

**Why is prefill compute-bound but decode memory-bandwidth-bound?** #flashcard
Prefill processes all S input tokens in parallel — arithmetic intensity scales with S, quickly exceeding the GPU's compute/bandwidth threshold. Decode generates one token at a time (batch=1): FLOPs=2N but bytes=N (full weight load per token), giving intensity ~1 op/byte, far below the threshold — bound by how fast weights stream from HBM.

**Why does decode throughput scale near-linearly with batch size, up to a point?** #flashcard
At batch B, FLOPs scale to 2NB while bytes loaded stay ~N (weights shared across the batch), so arithmetic intensity becomes B ops/byte. Below the compute threshold (~500 on A100), more batching is nearly free throughput; above it, decode becomes compute-bound and stops scaling.

**What does the KV cache store, and why is it needed?** #flashcard
It stores the Key and Value projections for every previous token position so autoregressive decoding doesn't recompute them at each step — recomputing from scratch would cost O(t²) total work vs O(t) with caching.

**Why can KV cache memory exceed model weight memory at high batch sizes?** #flashcard
KV cache scales with layers × KV heads × head_dim × context_length × batch_size × dtype_bytes, so on long-context, large-batch workloads it can dwarf the (fixed) model weight footprint — e.g. 100 concurrent 8K-context requests on an 8B model needs ~54GB of KV cache alone.

**What problem does PagedAttention solve, and how?** #flashcard
Pre-allocating a contiguous max-context block per request wastes 40-80% of KV memory to fragmentation on variable-length sequences. PagedAttention borrows OS-style virtual memory: fixed-size physical blocks mapped via a per-request logical→physical block table, allocated on demand and freed on completion — eliminating fragmentation and enabling 2-4x more concurrent requests.

**How does continuous batching differ from static batching, and why does it raise GPU utilization?** #flashcard
Static batching waits for the whole batch to finish before admitting new requests, so short requests idle waiting on the longest one. Continuous batching re-forms the batch at every decode step, immediately filling a freed slot with a queued request — raising GPU utilization from ~30% to >80%.

**Why does chunked prefill reduce TTFT, and what's the cost?** #flashcard
Splitting a long prompt's prefill into small chunks (e.g. 512 tokens) interleaved with decode steps lets the first output token stream after one chunk instead of the full prompt, cutting TTFT dramatically. Cost: TPOT increases slightly due to interleaving/context-switch overhead.

**How does speculative decoding speed up generation despite decode being memory-bound?** #flashcard
A small draft model generates K candidate tokens autoregressively (cheap); the large target model verifies all K in one parallel forward pass (which is compute-bound, so batching K tokens is nearly free) and accepts/rejects each via a probability ratio test — yielding multiple accepted tokens per expensive target-model call instead of one.

**Why route LLM requests to different model sizes rather than always using the largest model?** #flashcard
Most requests don't need the largest model's capability; routing by input complexity, user tier, or small-model confidence sends easy requests to cheap/fast models and only escalates hard ones — cutting average cost and latency without sacrificing quality on requests that need it.

**In LLM capacity planning, why is decode throughput (not total throughput) the binding constraint?** #flashcard
Prefill is compute-bound and processed once per request, but decode happens once per output token and is memory-bandwidth-bound — so GPU count is sized off `QPS × avg_output_tokens / decode_tokens_per_sec_per_GPU`, not off total (input+output) token volume.
