---
module: LLMs
topic: Revision Guide
subtopic: ""
status: unread
tags: [llms, ml, revision-guide]
---
# 10 LLMs — Revision Guide

Single-file revision pass over `10-llms/`: training pipeline, scaling, architecture, evaluation, stability, fine-tuning, context extension, KV cache, MoE, and tokenization. Use the outline to check you can place every concept in its tree, then work the graded Q&A to test recall and synthesis before an interview.

## Structure Overview

```markdown
- **10 LLMs**
  - **Training Process**
    - Pretraining: next-token pred
    - SFT: loss masking
    - RLHF: reward model + PPO
    - KL penalty vs SFT ref
    - DPO: implicit reward
    - ORPO: no ref model
    - Constitutional AI / RLAIF
    - Reward hacking
    - Continual pretraining
  - **Scaling & Data**
    - Kaplan scaling laws
    - Chinchilla: 20 tok/param
    - Compute vs inference-optimal
    - Emergent abilities debate
    - Data dedup: MinHash LSH
    - Synthetic data
    - Test-time compute: CoT/BoN/MCTS
    - Token scarcity
  - **Architecture**
    - Self-attention QKV
    - Multi-head attention
    - LayerNorm vs RMSNorm
    - SwiGLU FFN
    - Flash Attention
    - Sliding window attention
    - RoPE mechanics
    - Model family surveys
  - **Evaluation**
    - MMLU / BBH / ARC
    - HellaSwag / WinoGrande
    - GSM8K / MATH
    - HumanEval / MBPP pass@k
    - SWE-Bench
    - Needle-in-Haystack / RULER
    - TruthfulQA / BBQ / MT-Bench
    - BLEU/ROUGE/BERTScore/PPL
    - LLM-as-judge + bias
    - LMSYS Arena Elo
    - Contamination detection
    - RAGAS
  - **Training Stability**
    - Loss spikes: cause/detect/recover
    - Gradient clipping norm vs value
    - Warmup necessity
    - FP16 vs BF16 + loss scaling
    - Token/mode collapse detection
    - RLHF failures: KL explosion
    - Adaptive KL controller
    - DPO mode collapse -> IPO
    - ZeRO stages 1-3
    - Checkpoint averaging
    - Monitoring: grad_norm, clip_frac, KL
  - **Fine-Tuning at Scale**
    - Full FT memory budget ~16B/param
    - LoRA: W'=W0+ab/r BA
    - Rank selection
    - QLoRA: NF4 + double quant
    - GPTQ Hessian PTQ
    - AWQ activation-aware
    - GGUF k-quants
    - ZeRO-3 + CPU offload
    - Gradient checkpointing
    - Prefix tuning
    - Adapter layers
    - LoRA variants: DoRA/AdaLoRA/VeRA
  - **Context Extension**
    - RoPE failure at m>L_train
    - Linear interpolation
    - NTK-aware scaling
    - YaRN: ramp + temp correction
    - LongRoPE evolutionary search
    - ALiBi linear bias
    - StreamingLLM attention sinks
    - Lost-in-the-middle
    - Qwen3 iRoPE / NoPE
    - Sparse attn: Longformer/BigBird
    - Context compression: LLMLingua/Gist
    - Mamba / SSM alternative
  - **KV Cache / MQA / GQA**
    - MHA memory formula
    - MQA: 1 shared KV head
    - GQA: grouped KV heads
    - DeepSeek MLA latent compression
    - Prefix caching
    - KV quantization int8/int4
    - Llama-3 8B serving example
  - **MoE & Routing**
    - Sparse activation top-K
    - Softmax router
    - Expert choice routing
    - Load balance loss f_i * P_i
    - Expert collapse
    - DeepSeek shared+routed experts
    - Capacity factor / token dropping
    - Expert parallelism all-to-all
    - Expert specialization
    - Production models table
  - **Tokenization**
    - BPE algorithm
    - Byte-level BPE
    - WordPiece PMI merge
    - SentencePiece + Unigram LM
    - Tiktoken encodings
    - Vocab size tradeoffs
    - Softmax bottleneck
    - Fertility across languages
    - Number tokenization artifacts
    - Tokenizer-free: MegaByte/CANINE
```

## Questions

### Easy

1. What objective is a base LLM pretrained on, and why does that alone produce broad capabilities?
2. What does SFT change relative to a base model, and how is the loss masked?
3. Name the three stages of the classic RLHF pipeline.
4. What does the KL penalty term in RLHF constrain, and against what reference?
5. What is the core difference between DPO and RLHF in terms of models required?
6. State the Chinchilla compute-optimal ratio of tokens to parameters.
7. Why did Kaplan-era models end up systematically undertrained?
8. What is the difference between "compute-optimal" and "inference-optimal" training?
9. Define fertility in the context of tokenization.
10. Why does byte-level BPE eliminate `<UNK>` tokens entirely?
11. What is the `##` prefix in WordPiece used for?
12. What is the purpose of the `▁` marker in SentencePiece?
13. What is the softmax bottleneck at the output layer, and what two numbers determine its cost?
14. What is the difference between MHA, MQA, and GQA at a one-sentence level?
15. Write the KV cache memory formula in terms of layers, heads, head dim, sequence length, batch, and dtype bytes.
16. What does gradient clipping by norm do to a gradient vector, in one line?
17. Why is BF16 preferred over FP16 for LLM pretraining despite lower precision?
18. What is a loss spike, and name one common cause.
19. What is the purpose of LR warmup at the start of training?
20. Give the LoRA update formula and state which matrices are trained.
21. What does QLoRA add on top of LoRA?
22. What is the difference between GPTQ and AWQ at a high level?
23. Why does RoPE fail to extrapolate to positions beyond its training length?
24. What is the core idea of ALiBi, and how does it differ from RoPE?
25. What is "lost in the middle," and where does the model perform best in a long context?
26. In an MoE layer, what does "top-K routing" mean?
27. What is expert collapse, and what does it look like symptomatically?
28. What is the difference between MMLU and HellaSwag in what they measure?
29. What is perplexity, and what does a low value guarantee (and not guarantee)?
29. What is pass@k used to evaluate, and in what domain?
30. Name one known bias of LLM-as-judge evaluation.

### Medium

1. Why does next-token prediction alone induce reasoning, coding, and translation abilities without explicit supervision for any of them?
2. Compare DPO and ORPO: what does ORPO remove from the pipeline, and what's the tradeoff?
3. Why can too-aggressive RLHF fine-tuning degrade a model's pretrained capabilities even as reward score increases?
4. Explain why emergent abilities might be partly a measurement artifact rather than a true phase transition.
5. Walk through why Chinchilla-optimal training is not the same as cost-optimal serving for a model deployed at massive scale.
6. Why does deduplication (e.g. MinHash LSH) materially change effective compute-optimal token counts?
7. Compare test-time compute scaling (CoT, best-of-N, MCTS) to parameter/data scaling as two different axes of the same scaling law framework.
8. Derive why the 1/√d_k scaling in attention prevents softmax saturation, referencing the variance of the dot product.
9. Why do different attention heads specialize without any explicit supervision assigning them roles?
10. Explain why GQA is described as "interpolating" between MHA and MQA, and what happens at each extreme (G=H, G=1).
11. Walk through the KV cache memory savings of Llama-3 70B's GQA (8 KV heads vs 64 query heads) vs full MHA.
12. Explain how DeepSeek's MLA achieves greater compression than GQA while reportedly maintaining or exceeding MHA quality.
13. Why does the load-balancing loss use f_i (token fraction) times P_i (mean probability) instead of just penalizing f_i directly?
14. Explain the tradeoff of capacity_factor in MoE token dropping, and why training and inference typically use different values.
15. Compare expert-choice routing to softmax top-K routing: what does expert-choice guarantee, and what new problem does it introduce?
16. Why is "MoE has the same FLOPs as a 13B dense model" a misleading framing for total cost of ownership?
17. Explain why NTK-aware scaling improves on plain linear position interpolation for context extension.
18. Why does YaRN require an attention temperature correction in addition to the NTK-aware ramp function?
19. Compare StreamingLLM's attention sinks approach to true long-context recall — what capability does it not provide?
20. Why does Qwen3's iRoPE (dropping RoPE every 4th layer) improve extrapolation to very long contexts?
21. Explain why gradient clipping by norm is preferred over clipping by value in most LLM training setups.
22. Walk through why FP16 needs loss scaling but BF16 doesn't, referencing exponent vs mantissa bit allocation.
23. Explain the adaptive KL controller in RLHF: what does it adjust, and in response to what signal?
24. Why can DPO mode-collapse, and how does IPO's regularization address it?
25. Compare ZeRO-2 and ZeRO-3 in terms of what is sharded and the resulting communication/memory tradeoff.
26. Explain why LoRA rank selection interacts with "intrinsic dimensionality" of the fine-tuning task.
27. Walk through why QLoRA's NF4 format is well-suited to neural network weight distributions specifically.
28. Compare GPTQ's Hessian-based correction to AWQ's activation-aware approach: what does each preserve, and why might they diverge in accuracy?
29. Explain why LLM-as-judge tends to favor longer, more confident responses, and what that bias implies for using it to rank RLHF policies.
30. Why is contamination detection necessary for benchmarks like MMLU/GSM8K, and what's one practical detection method?
31. Explain why RAGAS's "faithfulness" and "context recall" are measuring different failure modes in a RAG pipeline.
32. Compare WordPiece's PMI-based merge criterion to BPE's frequency-based merge — what kind of subwords does each tend to produce differently?
33. Explain why tiktoken's cl100k_base splitting numbers into individual digits specifically helps arithmetic performance.

### Hard

1. Derive the expected number of forward passes to converge on the "compute-optimal" frontier, and explain why a fixed FLOPs budget implies a specific model-size/data-size split under Chinchilla.
2. A model shows a loss spike at step 40K that partially recovers within 500 steps but leaves a permanently higher loss floor. Walk through your full diagnostic and recovery procedure, including what data/checkpoint forensics you'd run.
3. You must choose between YaRN, LongRoPE, and ALiBi to extend a RoPE-pretrained 8K-context model to 256K. Walk through the decision tree given constraints on fine-tuning budget and target quality.
4. Derive the KV cache memory in bytes for a 70B model with GQA (8 KV heads, 128 head_dim, 80 layers, batch 32, 128K context, FP16) and explain what breaks first as context grows.
5. Explain precisely why DeepSeek MLA's latent compression can preserve or exceed MHA quality despite compressing KV to a much smaller latent — what's the mechanism that avoids the MQA-style quality loss?
6. Derive the load-balancing loss gradient path: since f_i is non-differentiable (argmax-derived), explain exactly how gradient flows through P_i to influence routing decisions, and why α is bounded in practice.
7. A production MoE model shows uniform per-expert token fractions but degraded downstream quality after a training resumption. Propose a diagnostic beyond the token-fraction histogram that could reveal a subtler routing failure.
8. Explain why RLHF's reward-hacking failure mode is a Goodhart's Law problem specifically, and construct a concrete example where a reward model overfit to human preferences would diverge from true task success.
9. Contrast the failure surfaces of full fine-tuning + ZeRO-3 + CPU offload vs LoRA + gradient checkpointing for fine-tuning a 70B model on a single 8x80GB node — where does each approach hit a wall first?
10. Derive why RoPE's dot-product-only-depends-on-(m-n) property breaks down numerically (not just theoretically) at positions far beyond training length, and explain what NTK-aware scaling changes about the frequency spectrum to mitigate it.
11. Explain the interaction between sliding-window attention (or StreamingLLM) and KV cache prefix caching in a serving system — what invalidates the cache, and what doesn't?
12. Walk through why "lost in the middle" persists even in models explicitly trained with long-context data, and propose an evaluation methodology (beyond Needle-in-a-Haystack) that would detect it more reliably.
13. Derive the total parameter memory for Mixtral 8x7B (46.7B total, 12.9B active) at FP16, and explain why the serving GPU count is set by total params while throughput is set by active FLOPs — construct the crossover point where a dense 13B model becomes more cost-efficient to serve.
14. A DPO run shows the implicit reward margin (log π_θ(y_w)/π_ref(y_w) − log π_θ(y_l)/π_ref(y_l)) increasing while validation win-rate against held-out human preferences decreases. Diagnose what's happening and how IPO's objective would change the outcome.
15. Explain precisely why ZeRO-3's parameter sharding requires an all-gather before every forward/backward compute step, and derive why this makes ZeRO-3 communication-bound in a way ZeRO-1 is not.
16. Construct a case where FP16 loss scaling with a dynamic scale factor still silently corrupts training (i.e., doesn't trigger an inf/nan skip) — what numerical regime causes this?
17. Explain why token-level fertility differences across languages translate into unequal effective compute allocation during pretraining, and propose a data-mixing correction.
18. Derive why increasing vocabulary size from 32K to 128K changes the FLOPs-optimal tokens-per-parameter ratio (via the embedding/output-head parameter count), and explain the implication for Chinchilla scaling law calculations at very large vocab sizes.
19. You observe that a speculative-decoding-style MTP head trained jointly with a base model changes the base model's own next-token loss curve during pretraining. Explain the plausible mechanism and how you'd isolate whether MTP is helping or hurting the primary objective.
20. Explain why GGUF's per-tensor mixed-precision k-quant scheme (e.g. Q4_K_M) can outperform a uniform 4-bit GPTQ quantization at the same average bits-per-weight, referencing which tensors are more sensitive to quantization error.
21. Walk through why the interaction between capacity_factor, expert parallelism degree, and all-to-all communication creates a nonlinear throughput cliff as batch size decreases in production MoE serving.
22. Derive why RMSNorm removing the mean-centering step (relative to LayerNorm) doesn't hurt training stability in practice, referencing what LayerNorm's re-centering was actually doing versus what RMSNorm's re-scaling preserves.
23. Explain a scenario where a model passes Needle-in-a-Haystack at 100% but still fails a real multi-hop long-document task — what capability gap does NIAH fail to test?
24. Derive why prefix caching's cache-hit rate degrades non-linearly as system-prompt diversity increases across concurrent requests, and propose a request-batching strategy to preserve hit rate.
25. Explain why combining QLoRA (4-bit NF4 base) with DoRA's magnitude-direction decomposition is nontrivial — what does quantization do to the weight-norm decomposition DoRA relies on?
26. Construct an argument for why "test-time compute" (CoT, best-of-N) scaling could eventually hit diminishing returns for the same underlying reason parameter scaling does, and what evidence would distinguish the two regimes.
27. Explain precisely how the adaptive KL controller can itself become unstable (oscillate) during RLHF, and propose a fix grounded in control theory intuition (e.g., PI vs PID-style controller).
28. Walk through what changes numerically in attention entropy and vocab coverage metrics in the specific case of RLHF-induced mode collapse versus SFT-induced overfitting — how would you distinguish the two from logs alone?
29. Derive why LongRoPE's evolutionary search for per-dimension scaling factors is necessary rather than a closed-form solution — what makes the loss landscape over per-frequency scale factors non-convex enough to require search?
30. Explain the failure mode where a model fine-tuned with LoRA on top of a model that was itself served in W4A16 quantization shows quality degradation at merge time — what does "merging LoRA into a quantized base" actually require, and why is naive merging lossy?
