---
module: LLM Applications
topic: Applications
subtopic: Speculative Decoding
status: unread
tags: [llms, ml, applications-speculative-decod]
---
# Speculative Decoding

---

## The Autoregressive Bottleneck

**The problem:** generating each token requires one full forward pass through all N layers of the target model. On a 70B model in bfloat16 (140GB of weights), each forward pass reads those 140GB from GPU high-bandwidth memory. At ~2TB/s HBM bandwidth on an A100, that is ~70ms per token — a hard floor imposed by memory bandwidth, not by the model's arithmetic. The GPU's tensor cores are largely idle, waiting for weights to arrive.

This is an arithmetic intensity problem: the ratio of floating-point operations to memory bytes loaded is too low at small batch sizes. The GPU is memory-bandwidth-bound, not compute-bound. Any technique that produces multiple tokens while reading the weights once converts memory-bandwidth utilization into arithmetic utilization — a pure win.

---

## Standard Speculative Decoding

**The problem:** we need to produce more than one token per weight-read cycle. But generating token t+2 requires knowing token t+1 first — the autoregressive dependency prevents parallelism across output tokens. The target model cannot generate multiple tokens independently.

**The core insight:** a small, fast draft model can cheaply propose K candidate tokens. The large target model can verify all K proposals in a single forward pass — the same cost as generating one token — because verification only requires computing the probability the target model assigns to each draft token, not sampling anew. If most proposals are correct, K tokens are produced for approximately the cost of one target model pass.

**The mechanics:**

Step 1 — Draft: run the small draft model autoregressively for K steps to produce K candidate tokens with their probabilities p₁...pₖ.

Step 2 — Verify: run the target model once on the full input + K draft tokens, obtaining the target model's probabilities q₁...qₖ for each position.

Step 3 — Accept/reject each draft token via rejection sampling (left to right):
```
For token i:
    acceptance probability = min(1, qᵢ(x̃ᵢ) / pᵢ(x̃ᵢ))
    sample u ~ Uniform[0, 1]
    if u ≤ acceptance_prob: ACCEPT x̃ᵢ, continue to i+1
    else: REJECT. Sample from adjusted distribution max(0, q - p) / Z. STOP.
```

Step 4 — Output all accepted tokens plus a bonus token (if all K were accepted, sample one more from the target model's distribution at position K+1).

**Correctness guarantee:** the rejection sampling step guarantees the output distribution is exactly the target model's distribution — this is not an approximation. The adjusted distribution on rejection, `max(0, q - p) / Z`, combined with already-accepted tokens, preserves the marginal probability of every output sequence. Speculative decoding is lossless.

**Expected tokens per target model call:**
```
If acceptance rate per token = α:
E[tokens] = (1 - α^{K+1}) / (1 - α)

At α=0.8, K=4:  E[tokens] = (1 - 0.8^5) / (1 - 0.8) ≈ 3.36
At α=0.9, K=4:  E[tokens] = (1 - 0.9^5) / (1 - 0.9) ≈ 4.10
```

**Speedup:** if the draft model is β times faster than the target, the speedup is approximately (E[tokens per step]) / (1 + K/β). At α=0.8, K=4, β=10: speedup ≈ 2.3×.

**What breaks:** speedup depends on acceptance rate, which depends on how well the draft model's distribution matches the target's. High-temperature sampling reduces acceptance rates. Long speculative chains (K > 8) rarely help — most proposals are rejected before reaching the end. Large batch sizes already have high GPU utilization — draft overhead without proportional speedup makes things worse. The draft model must share vocabulary with the target.

---

## Medusa — Multiple Draft Heads

**The problem:** standard speculative decoding requires a separate draft model — additional memory, additional deployment complexity, and a second model to maintain and update. If the draft model is outdated relative to the target model, acceptance rates fall.

**The core insight:** attach K additional prediction heads to the final transformer layer of the target model. Each head predicts a different future token position from the same hidden state. This eliminates the separate draft model entirely — drafting requires only K cheap linear layer forward passes on the already-computed target model hidden state.

**The mechanics:**
```
Transformer hidden state h_t
    → Head 0: standard LM head → logits for token t+1
    → Head 1: W_1 · h_t → logits for token t+2
    → Head 2: W_2 · h_t → logits for token t+3
    ...
    → Head K: W_K · h_t → logits for token t+K+1
```

Only the K new heads are trained; the base model is frozen. Tree attention generates multiple candidate paths (top-s from each head), and the target model verifies all paths in a single forward pass using a custom attention mask where each node attends only to its ancestors. The longest valid accepted prefix is selected.

**What breaks:** Medusa heads see only the current hidden state — no recurrent context about how the sequence has been developing. Acceptance rates (0.65–0.75) are lower than a trained draft model (0.80–0.90). The base model must be fine-tuned to train the heads, so Medusa cannot be applied to an arbitrary model without additional compute. Speedup (1.5–2.5×) is lower than standard speculative decoding (2–3×) when a good draft model is available.

---

## Eagle — Feature-Level Drafting

**The problem:** Medusa heads predict future tokens from the current hidden state alone. But the target model's next-token prediction is most accurate when computed from a hidden state that closely approximates the target model's actual hidden state at that position. A linear head predicting two positions ahead has no access to the intermediate computation the target model would have performed.

**The core insight:** train a lightweight single-layer transformer that takes both the current token embedding and the target model's hidden state as inputs, and predicts the target model's next hidden state. By conditioning on the target model's actual internal representations, the draft model's predictions are far better calibrated than Medusa's.

**The mechanics:**
```
Eagle draft model input: [token embedding at t] + [target model hidden state at t]
→ single transformer layer (the eagle model)
→ predicted hidden state ĥ_{t+1}
→ LM head (shared with target model)
→ draft token logits
```

The LM head is shared with the target model — no new output layer to train. The single transformer layer is small (~1/30 the compute of a full target model layer).

**What breaks:** Eagle requires access to the target model's hidden states during drafting, creating tighter coupling between draft and target than standard speculative decoding. The eagle model must be retrained for each target model. At high batch sizes where the GPU is already compute-bound, Eagle's overhead does not pay off.

**Eagle-2 — dynamic draft tree:** Eagle-2 makes the draft tree adaptive. High-confidence branches are expanded; low-confidence branches are pruned. This concentrates compute on candidate paths most likely to be accepted, achieving 3–4× speedup vs. 2–3× for base Eagle.

---

## Self-Speculative Decoding

**The problem:** both standard speculative decoding and Eagle require either a separate model or additional trained components. For memory-constrained deployments, even a small draft model may not fit.

**The core insight:** the target model itself can act as its own draft model by exiting early. The first L/2 layers produce a hidden state that often contains enough information to predict the correct next token. Run the early exit as the draft; verify with the full model.

**The mechanics:**
```
Standard:         pass through all L layers → token
Self-speculative: exit at layer L/2 → draft token
                  full pass through L layers → verify
```

If the early exit prediction matches the full model's prediction (acceptance rate ~0.7–0.8), effective token throughput increases without any additional memory.

**What breaks:** the early exit draft produces lower-quality candidates than a dedicated draft model, resulting in lower acceptance rates. Works best for deterministic tasks (code completion, factual retrieval) where the model is confident early in the forward pass. High-temperature or creative generation benefits less.

---

## Multi-Token Prediction

**The problem:** all of the above require either a separate model, additional trained heads, or architectural coupling. Can multi-token generation be built into the training objective itself?

**The core insight:** train auxiliary prediction heads alongside the standard LM head. Each auxiliary head predicts a token k positions ahead. At inference, the auxiliary heads generate speculative tokens automatically as a byproduct of the standard forward pass — no second model required.

**The mechanics (DeepSeek MTP):**
```
Forward pass → main head: predicts t+1 (standard)
            → aux head 1: predicts t+2 (speculative)
            → aux head 2: predicts t+3 (speculative)

If aux predictions match sequential decoding: accept and advance multiple positions
If rejected: fall back to one-token advance
```

The speculation cost is zero at inference — the auxiliary heads are computed anyway. The only cost is the training overhead of learning to predict multiple positions ahead.

**What breaks:** multi-token prediction auxiliary heads must be trained from scratch — this cannot be added to an existing model post-hoc without significant retraining. For very creative or high-entropy tasks, multi-token prediction provides minimal speedup because future tokens are genuinely uncertain.

---

## When Speculative Decoding Helps and When It Hurts

**The problem:** speculative decoding adds draft computation overhead. This overhead only pays off if the acceptance rate is high enough and the batch size is small enough that the GPU was previously underutilized.

**Conditions where it helps:**
- Small batch size (1–4): memory bandwidth is the bottleneck; speculative decoding multiplies tokens per weight-read.
- Long generation sequences (code, long documents): amortizes draft overhead over many accepted tokens.
- High-quality draft model from the same family (LLaMA 7B for LLaMA 70B): high acceptance rates.

**Conditions where it hurts:**
- Large batch sizes (32+): the GPU is compute-bound; speculative decoding adds coordination overhead without proportional speedup.
- Very short responses (< 20 tokens): draft overhead is not amortized.
- High-temperature sampling: widens the distribution gap between draft and target, lowering acceptance rates.

**Acceptance rate by task:**

| Task | Typical acceptance rate | Speedup |
|:---|:---|:---|
| Code completion | 0.85–0.92 | 2.5–4× |
| Factual QA | 0.75–0.85 | 2–3× |
| Creative writing | 0.60–0.75 | 1.5–2× |
| Math reasoning (extended CoT) | 0.55–0.70 | 1.3–1.8× |

Code generation benefits most because code has high local predictability — variable names repeat, boilerplate is formulaic. Creative generation and mathematical reasoning have many plausible continuations, so the draft model's prediction is less likely to match the target's.

**Break-even analysis:** if the draft model is β times cheaper per token, speculative decoding with K draft tokens breaks even when:
```
E[tokens per step] / (1 + K/β) ≥ 1

At K=4, β=10: break-even at α ≥ 1/4 + 1/10 = 0.35
```
Any reasonable draft model achieves α > 0.5 at small batch sizes. Speculative decoding almost always helps for single-user interactive use; it rarely helps for high-throughput batch inference.

*Related: [Inference Optimization](06-inference-optimization.md) | [Tuning and Optimization](10-tuning-optimization.md)*

## Flashcards

**Why does speculative decoding speed up generation without changing the output distribution?** #flashcard
A small draft model proposes K tokens; the target model verifies all K in a single forward pass (same cost as generating one token) and uses rejection sampling — accept token i with probability min(1, q_i/p_i), else resample from max(0, q-p)/Z. This guarantees the output distribution exactly matches the target model's, so speedup is "free" whenever the draft model's guesses are often correct.

**What determines the actual speedup from speculative decoding, and when does it hurt instead of help?** #flashcard
Speedup depends on acceptance rate α (how well draft and target distributions match) and draft-model speed ratio β: speedup ≈ E[tokens/step] / (1 + K/β). It helps at small batch sizes (memory-bandwidth-bound, GPU underutilized) and long generations (amortizes draft overhead); it hurts at large batch sizes (already compute-bound — adds overhead with no gain) and very short responses.

**Why does speculative decoding work better for code than for creative writing or math CoT?** #flashcard
Code has high local predictability (repeated variable names, boilerplate), so a small draft model's next-token guesses match the target's more often (acceptance ~0.85-0.92, 2.5-4× speedup). Creative writing and extended math reasoning have many equally plausible continuations, so draft/target disagree more often (acceptance ~0.55-0.75, 1.3-2× speedup).

**How does Medusa avoid needing a separate draft model, and what's the cost?** #flashcard
It attaches K extra linear prediction heads to the target model's final hidden state, each predicting a different future position, so drafting is just K cheap linear passes on already-computed hidden states — no second model to deploy. Tradeoff: heads see only the current hidden state (no recurrent context), so acceptance rates (0.65-0.75) and speedup (1.5-2.5×) are lower than a well-trained separate draft model, and the heads still require fine-tuning on the frozen base model.

**How does Eagle improve on Medusa, and what's the added coupling cost?** #flashcard
Eagle trains a small single-layer transformer that conditions on both the token embedding and the target model's actual hidden state to predict the target's next hidden state (sharing the target's LM head) — this is better calibrated than Medusa's context-free linear heads. Cost: tighter coupling to the target model (must be retrained per target) and it needs access to the target's internal hidden states during drafting.

**What is self-speculative decoding and when does it work best?** #flashcard
The target model drafts by exiting early (e.g., at layer L/2) and verifies with the full L-layer pass — no separate model or trained components needed at all. Works best on deterministic tasks (code completion, factual retrieval) where the model is already confident early in the forward pass; benefits less on high-temperature/creative generation.

**How does multi-token prediction (e.g., DeepSeek MTP) differ from Medusa/Eagle in when its cost is paid?** #flashcard
Auxiliary heads predicting t+2, t+3, etc. are trained jointly with the main LM head from scratch, so at inference the speculative tokens are a free byproduct of the standard forward pass — zero added inference cost. The cost is paid entirely during training (can't be bolted onto an existing model without retraining), and it still provides little benefit on high-entropy/creative tasks.
