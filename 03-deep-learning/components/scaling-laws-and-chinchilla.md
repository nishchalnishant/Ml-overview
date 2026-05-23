---
module: Deep Learning
topic: Components
subtopic: Scaling Laws And Chinchilla
status: unread
tags: [deeplearning, ml, components-scaling-laws-and-ch]
---
# Scaling Laws and Chinchilla

How model size, data, and compute interact. Every foundation model team makes decisions based on these laws. Critical for L5/L6 LLM interviews.

---

## 1. Kaplan Scaling Laws (2020)

**OpenAI's original finding (Kaplan et al.):** Loss follows a power law in N (parameters) and D (tokens):

$$L(N) = \left(\frac{N_c}{N}\right)^{\alpha_N}, \quad L(D) = \left(\frac{D_c}{D}\right)^{\alpha_D}$$

with α_N ≈ 0.076, α_D ≈ 0.095.

**Joint scaling:** For a fixed compute budget C ≈ 6ND FLOPs:

$$L(N, D) = \left(\frac{N_c}{N}\right)^{\alpha_N} + \left(\frac{D_c}{D}\right)^{\alpha_D} + L_\infty$$

**Kaplan's conclusion:** Model size is more important than data. Given fixed compute C, scale N as C^{0.73} and D as C^{0.27} — i.e., grow models faster than data.

**GPT-3 followed this:** 175B params, only ~300B tokens (≈1.7 tokens/param).

---

## 2. Chinchilla (2022)

**DeepMind's correction (Hoffmann et al.):** Kaplan's setup held training duration roughly constant across model sizes, which biased toward large models. With proper isoFLOP analysis:

$$\boxed{N_{opt} \propto C^{0.5}, \quad D_{opt} \propto C^{0.5}}$$

**The "20× rule":**
$$D_{opt} \approx 20 \cdot N_{opt}$$

**Table: compute-optimal model sizes**

| Compute (FLOPs) | Optimal N | Optimal D |
|---|---|---|
| 10²³ | ~400M | ~8B tokens |
| 10²⁴ | ~1B | ~20B tokens |
| 10²⁵ | ~4B | ~75B tokens |
| 10²³·⁵ (Chinchilla ~= GPT-3 budget) | ~67B | ~1.4T tokens |

**Chinchilla (70B, 1.4T tokens) outperformed Gopher (280B, 300B tokens)** on most benchmarks, despite 4× fewer parameters, because it was trained on 4.7× more data.

---

## 3. Chinchilla Math Derivation

**IsoFLOP setup:** Minimize L subject to C = 6ND = constant.

Using the empirical loss function:
$$L(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}$$

where E = irreducible entropy, A/B/α/β are fit constants.

**Minimize L(N) with D = C/(6N):**

$$\frac{\partial L}{\partial N} = -\frac{\alpha A}{N^{\alpha+1}} + \frac{\beta B}{D^{\beta+1}} \cdot \frac{C}{6N^2} = 0$$

Solving: $N_{opt} \propto C^{\beta/(\alpha+\beta)}$, $D_{opt} \propto C^{\alpha/(\alpha+\beta)}$.

With Chinchilla's fitted α ≈ 0.34, β ≈ 0.28:
- $N_{opt} \propto C^{0.45} \approx C^{0.5}$
- $D_{opt} \propto C^{0.55} \approx C^{0.5}$

→ Equal allocation to parameters and data.

---

## 4. Post-Chinchilla Revisions

**Llama 1 (Meta, 2023):** Trained 7B on 1T tokens (≈143 tokens/param, 7× over-trained vs Chinchilla). **Why?** Inference cost matters. A compute-optimal model is optimal for training compute only. If you'll serve the model 1 billion times, a smaller model that needs more training tokens is cheaper overall.

**Revised objective (inference-optimal):**

| Goal | Optimal strategy |
|---|---|
| Minimize training FLOPs | Follow Chinchilla: D ≈ 20N |
| Minimize inference cost at fixed quality | Over-train smaller model: D >> 20N |
| Minimize total cost (train + serve) | Depends on total inference volume |

**Rule of thumb (post-Llama 1):** Train a model that is 2–5× smaller than Chinchilla-optimal, but 5–10× more tokens. The model will match Chinchilla performance but run much faster at inference.

**Example:** Llama-3 8B trained on 15T tokens (1875 tokens/param vs 20 = Chinchilla). This is ~94× over-trained, but achieves competitive quality with smaller serving cost.

---

## 5. Test-Time Compute Scaling

**New paradigm (2024–2025):** Beyond training-time scaling.

**The idea:** Instead of only scaling pre-training, scale compute at inference time (test-time compute).

**Methods:**
1. **Chain-of-thought / scratchpad:** Let model "think" before answering → O(k) more tokens → better reasoning
2. **Best-of-N sampling:** Sample N responses, select best by reward model → quality improves as √N samples
3. **Process reward models (PRM):** Score intermediate reasoning steps, not just final answer
4. **Tree search (MCTS):** Explore multiple reasoning paths, prune by PRM → OpenAI o1/o3 style

**Empirical scaling:**
$$\text{Performance}(k) \approx a \cdot \log(k) + b$$

where k is the number of tokens/samples used at test time.

**Best-of-N accuracy:**
$$P(\text{at least one correct in N}) = 1 - (1 - p)^N$$

For p=0.3 (hard math problem), N=32: accuracy = 1 - 0.7^32 ≈ 99.9%.

```python
def best_of_n_accuracy(p_single: float, n: int) -> float:
    """Probability that at least one of N samples is correct."""
    return 1 - (1 - p_single) ** n

# OpenAI o1 insight: train model to generate long CoT before answering
# Test-time compute budget → model learns to use it efficiently
```

**Training-time vs test-time scaling trade-off:**

| Approach | Cost | Latency | Quality | Notes |
|---|---|---|---|---|
| Larger model | High (one-time) | Fixed | Better avg | Training-time |
| More tokens trained | Medium (one-time) | Fixed | Better avg | Training-time |
| Best-of-N | Linear in N | N× slower | Better worst-case | Test-time |
| Chain-of-thought | None | k× slower | Better reasoning | Test-time (prompt) |
| MCTS + PRM | Medium (train PRM) | Variable | Best reasoning | Test-time |

---

## 6. Data Quality vs Quantity

**FineWeb / DCLM finding:** Data quality dominates beyond a threshold.

**Quality filtering pipeline:**
```
Common Crawl (~100T tokens raw)
       │
  URL filtering (blocklists, adult content)
       │
  Language detection (fastText, keep English)
       │
  Quality filtering:
    - C4 heuristic: remove docs with <3 sentences
    - Perplexity filtering: train small LM, keep moderate-perplexity docs
    - Deduplication: MinHash at document and ngram level
       │
  ~5–15T high-quality tokens
```

**Contamination check:** If benchmark test data is in training set, eval scores are inflated. Use n-gram matching (13-gram overlap) to detect and remove.

**Data mix (Llama-3 example):**
| Source | Weight |
|---|---|
| Web (CommonCrawl filtered) | 50% |
| Code | 25% |
| Math | 10% |
| Books | 10% |
| Multilingual | 5% |

Code + math upsampling improves reasoning beyond their natural frequency in web data.

---

## 7. Emergent Capabilities

**Observation (Wei et al. 2022):** Some capabilities appear abruptly at scale thresholds, not smoothly.

```
Performance
    │
    │                        ▲
    │                       /
    │           ___________/
    │__________/
    └────────────────────── Model size
              ↑
         "emergence point"
```

**Debated:** Some argue emergence is an artifact of coarse metrics (accuracy vs log-loss shows smooth scaling). But certain capabilities (multi-step arithmetic, chain-of-thought) still appear to have genuine phase transitions.

**GPT-3 175B emergent behaviors:**
- Few-shot learning (not present at 1B)
- Chain-of-thought reasoning (not present at 7B)
- Code generation

---

## 8. Compute Efficiency Summary

**FLOPs for a forward pass of an N-parameter Transformer:**
$$\text{FLOPs} \approx 2N \text{ per token}$$

**FLOPs for a training step (forward + backward):**
$$\text{FLOPs} \approx 6N \text{ per token}$$

(Factor of 6 = 2 forward + 4 backward, where gradient of weights ≈ 2× forward.)

**Training compute:**
$$C = 6 \cdot N \cdot D \text{ FLOPs}$$

**Example:** GPT-3 training
- N = 175B, D = 300B tokens
- C = 6 × 175B × 300B = 315 × 10²¹ FLOPs ≈ 3.1 × 10²³ FLOPs
- At 312 TFLOP/s on V100 FP16: ≈ 10⁹ GPU-seconds ≈ 32 GPU-years

---

## Canonical Interview Q&As

**Q: What does Chinchilla say about GPT-3?**  
A: GPT-3 (175B params, 300B tokens, ≈1.7 tokens/param) was significantly under-trained relative to Chinchilla's compute-optimal allocation (~20 tokens/param). The same compute budget should have been allocated to a ~70B model trained on ~1.4T tokens. DeepMind demonstrated this with Chinchilla (70B, 1.4T tokens), which outperformed GPT-3 despite 2.5× fewer parameters. GPT-3 prioritized training-compute efficiency over inference efficiency — in 2020, inference cost was less of a concern than training cost.

**Q: Why do companies still train over-sized models if Chinchilla says smaller is compute-optimal?**  
A: Chinchilla is optimal for training FLOPs only. Once you account for inference cost — a 7B model deployed at 1B QPS costs far less than a 70B model — the economics change. If you'll serve a model 100B times, spending 3× more compute training a 3× smaller model saves money overall. Llama-3 8B (15T tokens) vs Chinchilla prescription (~160B tokens at 8B) is the extreme example: 94× over-trained, but inference cost dominates at Meta's scale.

**Q: What is test-time compute scaling and when does it help vs scaling parameters?**  
A: Test-time compute scaling allocates more FLOPs at inference (longer reasoning chains, sampling multiple responses, tree search) instead of at training. It helps most for reasoning tasks where: (1) the answer can be verified (math, code), (2) the model already has the capability but needs more steps to express it, (3) you can build a reward/process model to score intermediate steps. It doesn't help for: factual recall (you can't reason to an unknown fact), tasks requiring broad world knowledge, or ultra-low-latency serving. The key insight from o1: intelligence is not only a function of parameter count but also of computation budget at inference.

**Q: How do you estimate the compute budget for training a 13B model on 2T tokens?**  
A: C = 6 × N × D = 6 × 13B × 2T = 156 × 10²¹ ≈ 1.56 × 10²³ FLOPs. At A100 peak throughput of ~300 TFLOP/s (bf16, accounting for MFU ≈ 0.4 → effective ~120 TFLOP/s): time = 1.56 × 10²³ / (120 × 10¹²) ≈ 1.3 × 10⁹ GPU-seconds = 15K GPU-days ≈ 42 GPU-years. At $2/GPU-hour on 8× A100 nodes: cost ≈ $2.5M. Rule of thumb: 10²³ FLOPs ≈ $1M at current cloud GPU prices.

**Q: What are "emergent capabilities" and are they real?**  
A: Emergent capabilities appear to arise discontinuously at scale thresholds — absent at 7B, present at 70B. Examples: multi-step arithmetic, BIG-bench tasks, chain-of-thought. The debate: Schaeffer et al. (2023) argue emergence is a metric artifact — coarse accuracy metrics create step functions even when log-loss scales smoothly. With continuous metrics, many "emergent" capabilities show gradual improvement. The practical implication: don't assume capabilities are absent until a threshold — use continuous probing metrics, and test your specific capability of interest across model scales.

## Flashcards

**$N_{opt} \propto C^{0.45} \approx C^{0.5}$?** #flashcard
$N_{opt} \propto C^{0.45} \approx C^{0.5}$

**$D_{opt} \propto C^{0.55} \approx C^{0.5}$?** #flashcard
$D_{opt} \propto C^{0.55} \approx C^{0.5}$

**C4 heuristic?** #flashcard
remove docs with <3 sentences

**Perplexity filtering?** #flashcard
train small LM, keep moderate-perplexity docs

**Deduplication?** #flashcard
MinHash at document and ngram level

**Few-shot learning (not present at 1B)?** #flashcard
Few-shot learning (not present at 1B)

**Chain-of-thought reasoning (not present at 7B)?** #flashcard
Chain-of-thought reasoning (not present at 7B)

**Code generation?** #flashcard
Code generation

**N = 175B, D = 300B tokens?** #flashcard
N = 175B, D = 300B tokens

**C = 6 × 175B × 300B = 315 × 10²¹ FLOPs ≈ 3.1 × 10²³ FLOPs?** #flashcard
C = 6 × 175B × 300B = 315 × 10²¹ FLOPs ≈ 3.1 × 10²³ FLOPs

**At 312 TFLOP/s on V100 FP16?** #flashcard
≈ 10⁹ GPU-seconds ≈ 32 GPU-years
