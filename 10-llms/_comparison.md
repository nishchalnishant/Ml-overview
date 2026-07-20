---
module: LLMs
topic: Cheatsheet
subtopic: Comparison
status: reference
tags: [llms, ml, cheatsheet, comparison, rapid-review]
---
# LLM Techniques — Exhaustive Comparison Cheat Sheet

Standalone rapid pre-interview review. Every technique documented in `05-llms/`, grouped by family, with what-it-is / pros / cons / when-to-pick-this / key formula.

---

## 1. Alignment Methods (RLHF vs DPO vs ORPO vs PPO vs Constitutional AI)

- **RLHF (Reward Model + PPO)**: train a reward model on human preference pairs, then optimize the policy against it with PPO under a KL penalty to the reference model.
  - Pros: decouples preference modeling from policy optimization; reward model can be reused/audited; strong empirical track record (GPT-4, Claude).
  - Cons: 3 models in memory (policy, reward, reference) at once; PPO is unstable, hyperparameter-sensitive; reward hacking; expensive multi-stage pipeline.
  - Pick over DPO when: you need a reusable, inspectable reward signal, or want online exploration beyond the static preference dataset.
  - Formula: `L = E[r_θ(x,y)] − β·KL(π || π_ref)`; reward model trained via Bradley-Terry loss on pairwise comparisons.

- **DPO (Direct Preference Optimization)**: derives a closed-form reward implicit in the policy, so preference data can directly optimize the policy without a separate reward model or RL loop.
  - Pros: no reward model, no PPO, no RL instability; single-stage supervised-style loss; cheaper and more stable than RLHF.
  - Cons: fully offline — no exploration beyond the fixed preference dataset; sensitive to reference-model quality; can mode-collapse if preference pairs are low-diversity.
  - Pick over RLHF when: compute/engineering budget is limited, or the preference dataset is static and high-quality.
  - Formula: `L_DPO = -E[log σ(β log(π_θ(y_w|x)/π_ref(y_w|x)) − β log(π_θ(y_l|x)/π_ref(y_l|x)))]`.

- **ORPO (Odds Ratio Preference Optimization)**: combines SFT and preference alignment into one loss, penalizing the odds ratio of rejected vs chosen without any reference model.
  - Pros: no reference model needed (saves memory); single training stage merges SFT + alignment; simpler pipeline than DPO.
  - Cons: newer, less battle-tested; odds-ratio penalty term is a weaker signal than full DPO's log-ratio in some tasks.
  - Pick over DPO when: memory-constrained (no room for a frozen reference copy) or want to skip a separate SFT stage.

- **Constitutional AI / RLAIF**: replace human preference labels with AI-generated critiques/labels against a written constitution, then run standard RLHF/DPO on the AI-labeled data.
  - Pros: scales alignment without proportional human-labeling cost; constitution is auditable/editable.
  - Cons: quality bounded by the critiquing model's own alignment; can propagate the base model's blind spots.
  - Pick over human-labeled RLHF when: human preference data is the bottleneck and a capable-enough critiquing model exists.

**Comparison table**

| Method | Best for | Avoid when | Key tradeoff |
|:---|:---|:---|:---|
| RLHF (PPO) | Reusable reward signal, online exploration | Limited eng/compute budget | Stability/complexity vs flexibility |
| DPO | Fast, stable, offline alignment | Need online exploration | Simplicity vs no exploration |
| ORPO | Memory-constrained, single-stage pipeline | Need strongest possible alignment signal | Simplicity vs signal strength |
| Constitutional AI/RLAIF | Scaling beyond human labels | No capable critiquing model available | Scale vs inherited bias |

Failure modes: reward hacking (Goodhart's Law — optimizing the proxy reward diverges from true intent), KL explosion in RLHF, DPO mode collapse (IPO variant addresses this by regularizing the implicit reward gap).

---

## 2. Scaling Laws (Kaplan vs Chinchilla)

- **Kaplan Scaling Laws (2020)**: power-law relationship between loss and model size/data/compute; concluded model size should scale faster than data.
  - Pros: first rigorous large-scale scaling study; established that scaling reliably reduces loss.
  - Cons: fixed LR schedule and undertrained small models biased conclusions; led to systematically undertrained large models (e.g., original GPT-3-era models).
  - Pick over Chinchilla when: never in practice today — superseded.

- **Chinchilla Scaling Laws (2022)**: compute-optimal training requires scaling model size and data roughly equally (~20 tokens/parameter), not model size dominant.
  - Pros: corrected Kaplan's bias; showed smaller, more-trained models (Chinchilla 70B) beat larger undertrained ones (Gopher 280B); now the default heuristic.
  - Cons: "compute-optimal" ≠ "inference-optimal" — for models served at massive scale, overtraining a smaller model beyond Chinchilla-optimal on more tokens minimizes total (train+serve) cost.
  - Pick over Kaplan when: always, for training a new model from scratch — this is the default.

**Comparison table**

| Approach | Best for | Avoid when | Key tradeoff |
|:---|:---|:---|:---|
| Kaplan | Historical reference only | Any real training run today | Superseded, undertrains large models |
| Chinchilla (compute-optimal) | Minimizing train-time loss per FLOP | Model will be served at massive inference scale | Train-optimal, not serve-optimal |
| Inference-optimal (overtrained) | High-volume production serving | Compute-constrained one-off training | More tokens/param than Chinchilla to cut serving cost |

Related: emergent abilities are largely a measurement artifact of discontinuous metrics (e.g. exact-match) rather than true phase transitions; test-time compute scaling (CoT, best-of-N, MCTS) is now a third scaling axis alongside params and data.

---

## 3. Attention Variants (MHA vs MQA vs GQA vs MLA)

- **MHA (Multi-Head Attention)**: each of H heads has its own K and V projections.
  - Pros: maximum representational capacity per head; best quality ceiling.
  - Cons: KV cache scales with full H heads — largest memory footprint, most memory-bandwidth-bound at inference.
  - Pick over MQA/GQA when: quality is paramount and serving memory/throughput is not the bottleneck (e.g. research, small-scale serving).
  - KV cache size: `2 × L × H × d_head × seq_len × batch × bytes`.

- **MQA (Multi-Query Attention)**: all H query heads share a single K/V head.
  - Pros: KV cache shrinks by factor H — large memory/throughput win, best for high-batch serving.
  - Cons: noticeable quality degradation vs MHA; single shared K/V is a representational bottleneck.
  - Pick over GQA when: memory is the dominant constraint and some quality loss is acceptable (rare in practice now — GQA usually preferred).

- **GQA (Grouped-Query Attention)**: query heads are split into G groups, each group shares one K/V head (interpolates between MHA at G=H and MQA at G=1).
  - Pros: most of MQA's memory savings with most of MHA's quality; the modern default (LLaMA-2/3, Mistral).
  - Cons: still a design choice (choice of G) — not free; some quality gap vs full MHA remains.
  - Pick over MHA/MQA when: default choice for any new production model — best quality/memory tradeoff.
  - Llama-3 70B: GQA with 8 KV heads vs 64 query heads → 8x KV cache reduction vs MHA.

- **MLA (Multi-Head Latent Attention, DeepSeek-V2)**: compresses K and V into a shared low-rank latent vector per token, decompressed on the fly per head; also decouples a small RoPE-carrying component from the compressed content component.
  - Pros: far larger KV cache compression than GQA (DeepSeek reports ~13.5x vs standard MHA) while matching or exceeding MHA quality; enables much longer effective context at fixed memory.
  - Cons: significantly more complex to implement; the low-rank compression + RoPE-decoupling design is nontrivial; less plug-and-play than GQA.
  - Pick over GQA when: aggressive KV cache compression is required (very long context, high concurrency) and engineering investment is justified — this is DeepSeek-V2/V3's approach.

**Comparison table**

| Variant | Best for | Avoid when | Key tradeoff |
|:---|:---|:---|:---|
| MHA | Max quality, small-scale serving | High-throughput/long-context serving | Quality vs memory |
| MQA | Extreme memory constraints | Quality-sensitive tasks | Max compression, biggest quality hit |
| GQA | Default production choice | Need MLA-level compression | Balanced default |
| MLA | Long-context, high-concurrency serving at top quality | Simplicity/engineering time is limited | Best compression+quality, most complex |

---

## 4. Positional Encoding & Context Extension (RoPE vs ALiBi vs Learned vs Sinusoidal, + extension methods)

- **Sinusoidal (original Transformer)**: fixed, non-learned sin/cos functions of position, added to embeddings.
  - Pros: no learned parameters; deterministic; some extrapolation capacity by design intent.
  - Cons: in practice extrapolates poorly beyond training length; largely superseded.
  - Pick over RoPE/ALiBi when: rarely today — mostly historical/encoder models.

- **Learned absolute positional embeddings (GPT-2/3 style)**: a trainable embedding vector per position index.
  - Pros: simple, flexible, can fit training distribution well.
  - Cons: cannot extrapolate at all beyond the max trained position (no embedding exists for unseen positions); every new context length requires retraining.
  - Pick over RoPE/ALiBi when: fixed, known-in-advance max context length and no extrapolation needed.

- **RoPE (Rotary Positional Embeddings)**: encodes position by rotating query/key vectors in 2D subspaces by an angle proportional to position; relative position naturally falls out of the dot product.
  - Pros: encodes relative position implicitly; strong empirical performance; the modern default (LLaMA, Mistral, Qwen, GPT-NeoX).
  - Cons: extrapolation beyond trained context length fails without modification (attention scores distort at unseen positions) — this is exactly why NTK/YaRN/LongRoPE exist.
  - Pick over ALiBi when: want the strongest short-context quality and plan to use explicit extension techniques for long context.

- **ALiBi (Attention with Linear Biases)**: adds a static, non-learned linear penalty to attention scores proportional to the distance between query and key positions — no positional embeddings at all.
  - Pros: much better native extrapolation than RoPE/sinusoidal without any modification; simpler, cheaper (no rotation computation).
  - Cons: quality on in-distribution (short) context can lag RoPE-based models; less widely adopted in latest frontier models than RoPE.
  - Pick over RoPE when: extrapolation to unseen lengths without any fine-tuning/scaling trick is the priority.

**RoPE Extension methods (all address RoPE's extrapolation failure at unseen positions):**

- **Linear (Position) Interpolation**: compress position indices by the ratio of target/original context length so they fall within the trained range. Cheap, but degrades resolution — nearby tokens become harder to distinguish; needs brief fine-tuning.
- **NTK-aware scaling**: scale RoPE's base frequency non-uniformly (high frequencies less compressed, low frequencies more) so local relationships stay sharp while long-range relationships stretch. Better quality retention than plain linear interpolation without fine-tuning, but not as strong as YaRN at extreme extension ratios.
- **YaRN**: combines NTK-aware interpolation with attention temperature scaling; strong at large extension factors (e.g. 4K → 128K) with minimal fine-tuning. More complex to tune (multiple hyperparameters) than plain NTK.
- **LongRoPE**: search-based per-dimension scaling factors, extends context to 2M+ tokens progressively (fine-tune at intermediate lengths before reaching target). Most extreme extension, but requires an expensive search + multi-stage fine-tuning process.
- **StreamingLLM**: keep a small number of initial "attention sink" tokens plus a sliding window of recent tokens, discard the rest — enables effectively infinite streaming generation without retraining. Does not give true long-range recall — only recency + sink tokens survive.

**Comparison table**

| Method | Best for | Avoid when | Key tradeoff |
|:---|:---|:---|:---|
| Sinusoidal/Learned | Historical/fixed-length models | Any length flexibility needed | No extrapolation |
| RoPE | Default modern base | Need extrapolation with zero tuning | Best short-context, needs extension tricks |
| ALiBi | Native extrapolation, simplicity | Squeezing max in-distribution quality | Simplicity/extrapolation vs peak quality |
| Linear Interpolation | Cheap quick extension | High extension ratios (>4x) | Cheap but degrades resolution |
| NTK-aware | Moderate extension, no/light fine-tune | Extreme extension ratios | Better than linear, worse than YaRN |
| YaRN | Large extension (4K→128K) with light FT | Simplicity is priority | Strong quality, more tuning knobs |
| LongRoPE | Extreme extension (millions of tokens) | Limited compute for staged fine-tuning | Max length, most expensive pipeline |
| StreamingLLM | Infinite streaming generation | True long-range recall needed | Infinite length, no real long-range memory |

Other long-context tools: Lost-in-the-Middle (U-shaped recall — put critical info first/last), Qwen3 iRoPE/NoPE (omit positional encoding every 4th layer to improve length generalization), sparse attention (Longformer, BigBird — local+global attention patterns), context compression (LLMLingua, Gist Tokens — compress prompt before feeding to model), Mamba/SSM (state-space models — linear-time sequence modeling, alternative to attention entirely for long sequences).

---

## 5. Fine-tuning / PEFT Methods (Full FT vs LoRA vs QLoRA vs Adapters vs Prefix/Prompt Tuning)

- **Full Fine-Tuning**: update all model parameters.
  - Pros: highest capacity to adapt, best ceiling on task performance.
  - Cons: full optimizer state (AdamW: 2x params in FP32 moments) + gradients + weights → ~16-20x param count in bytes without ZeRO; requires ZeRO stage 2/3 or multi-GPU sharding at scale; one full copy of weights per task (expensive to store/serve many variants).
  - Pick over PEFT when: task is far from pretraining distribution, or maximum quality justifies the compute/storage cost.

- **LoRA (Low-Rank Adaptation)**: freeze base weights, inject trainable low-rank decomposition matrices (A, B) into attention/MLP projections; `ΔW = BA`, rank r ≪ d.
  - Pros: drastically fewer trainable params (often <1%); small adapter checkpoints (MBs not GBs); no inference latency cost if merged into base weights; multiple task adapters can be swapped cheaply.
  - Cons: capacity bounded by rank r — may underfit for tasks far from base distribution; rank selection is a hyperparameter that needs tuning.
  - Pick over full FT when: task is reasonably close to base model's distribution, storage/serving multiple variants matters, or compute budget is limited.
  - Formula: `h = Wx + BAx`, where `B∈R^{d×r}`, `A∈R^{r×k}`, only A,B trained.

- **QLoRA**: LoRA on top of a 4-bit quantized (NF4) frozen base model, with double quantization and paged optimizers.
  - Pros: fine-tune massive models (65B+) on a single consumer/prosumer GPU; near full-precision-LoRA quality despite 4-bit base.
  - Cons: quantization/dequantization overhead slows training somewhat; slightly more complex tooling (bitsandbytes dependency).
  - Pick over LoRA when: GPU memory is the binding constraint (e.g., fine-tuning 70B on a single 24-48GB GPU).

- **Prefix Tuning**: prepend trainable continuous "virtual token" vectors to the keys/values at every layer, base model frozen.
  - Pros: very small number of trainable params; no change to model architecture/weights.
  - Cons: consumes context length budget; harder to optimize than LoRA (training can be unstable); generally lower quality ceiling than LoRA.
  - Pick over LoRA when: rarely today — LoRA generally dominates in the modern PEFT landscape; prefix tuning is more historical/niche.

- **Prompt Tuning**: like prefix tuning but only prepends trainable tokens at the input embedding layer (not every layer).
  - Pros: fewest trainable parameters of any PEFT method; extremely cheap.
  - Cons: lowest capacity/quality ceiling; needs a fairly large base model to work well.
  - Pick over prefix tuning/LoRA when: extreme parameter efficiency needed and base model is very large (capacity compensates).

- **Adapters**: insert small trainable bottleneck MLP layers between existing frozen transformer layers.
  - Pros: modular — can stack/combine adapters for multi-task; well-established technique predating LoRA.
  - Cons: adds inference latency (extra layers in the forward path, unlike LoRA which can be merged); generally now superseded by LoRA in practice.
  - Pick over LoRA when: need composable/stackable task modules where merge-into-base isn't required.

- **LoRA variants**: **LoRA+** (different LR for A and B matrices, faster convergence), **DoRA** (decomposes weight into magnitude+direction, applies LoRA to direction only — closer to full FT quality), **AdaLoRA** (adaptively allocates rank budget per layer based on importance), **VeRA** (shares frozen random A/B across layers, trains only small scaling vectors — even fewer params than LoRA), **LoRAMoE** (mixture of multiple LoRA experts, routed).

**Comparison table**

| Method | Best for | Avoid when | Key tradeoff |
|:---|:---|:---|:---|
| Full FT | Max quality, large distribution shift | Limited compute/storage, many task variants | Best ceiling, highest cost |
| LoRA | Default PEFT choice | Task very far from base distribution | Efficiency vs capacity (rank-bounded) |
| QLoRA | Single-GPU fine-tuning of huge models | Training speed is critical | Memory savings vs training overhead |
| Prefix/Prompt Tuning | Extreme parameter efficiency niche cases | Need strong quality | Cheapest, lowest ceiling |
| Adapters | Composable multi-task modules | Merge-free zero-latency inference needed | Modularity vs added inference latency |
| DoRA/AdaLoRA/VeRA | Squeezing more quality/efficiency from LoRA | Simplicity is preferred | Marginal gains vs added complexity |

---

## 6. Quantization (GPTQ vs AWQ vs GGUF vs bitsandbytes/NF4, + inference W4A16 vs W8A8)

- **GPTQ**: post-training quantization using layer-wise second-order (Hessian-based) error correction — quantizes weights one layer at a time, compensating remaining weights for the error introduced.
  - Pros: strong accuracy at 4-bit with no retraining; widely supported serving format; good for GPU inference.
  - Cons: quantization process itself is compute/time-intensive (calibration data + per-layer optimization); primarily weight-only (activations stay higher precision) — W4A16.
  - Pick over AWQ when: need broadest tooling/ecosystem support (very mature, many pre-quantized checkpoints available).

- **AWQ (Activation-aware Weight Quantization)**: identifies the small subset of weights most salient to activations (via activation magnitude statistics) and preserves their precision disproportionately, quantizing the rest aggressively.
  - Pros: often better accuracy retention than GPTQ at same bit-width, especially for instruction-tuned models; faster quantization process than GPTQ (no per-layer Hessian optimization).
  - Cons: slightly less mature ecosystem than GPTQ at time of writing; still weight-only quantization.
  - Pick over GPTQ when: accuracy at 4-bit is the top priority and the serving stack supports AWQ (vLLM, TGI both do).

- **GGUF (llama.cpp format)**: a serialization format (successor to GGML) supporting multiple quantization schemes (Q4_0, Q4_K_M, Q5_K_M, Q8_0, etc.), designed for CPU/edge/consumer-GPU inference.
  - Pros: runs well on CPU and consumer hardware (Apple Silicon, low-VRAM GPUs); huge ecosystem (llama.cpp, Ollama, LM Studio); flexible mixed-precision quant schemes per tensor type.
  - Cons: generally not the fastest option on datacenter GPUs vs GPTQ/AWQ+vLLM; quantization quality varies across scheme variants (K-quants vs legacy).
  - Pick over GPTQ/AWQ when: target deployment is CPU, edge, or consumer GPU rather than datacenter GPU serving.

- **bitsandbytes / NF4 (used in QLoRA)**: NormalFloat4 — a 4-bit data type whose quantization bins are chosen to match the expected normal distribution of neural net weights (information-theoretically optimal for normally distributed data), plus double quantization of the quantization constants themselves.
  - Pros: near-lossless 4-bit quantization tailored specifically for fine-tuning workflows; integrates directly into HF `transformers`/PEFT.
  - Cons: primarily designed for training (QLoRA) rather than max-throughput inference serving; not as inference-speed-optimized as GPTQ/AWQ+vLLM.
  - Pick over GPTQ/AWQ when: the goal is fine-tuning a quantized base (QLoRA), not pure inference serving.

- **Inference-time: W4A16 vs W8A8**: W4A16 keeps weights at 4-bit but activations at 16-bit (GPTQ/AWQ default) — memory-bound workloads benefit most since weight loading dominates. W8A8 (e.g. SmoothQuant) quantizes both weights and activations to 8-bit, enabling INT8 tensor core throughput — benefits compute-bound (large batch) workloads more.
  - Pick W4A16 when: memory-bandwidth-bound (small batch, long context). Pick W8A8/SmoothQuant when: compute-bound (large batch serving) and INT8 hardware paths are available.

**Comparison table**

| Method | Best for | Avoid when | Key tradeoff |
|:---|:---|:---|:---|
| GPTQ | Mature GPU serving, broad checkpoint availability | Very tight accuracy requirements at 4-bit | Ecosystem maturity vs AWQ's accuracy edge |
| AWQ | Best 4-bit accuracy on GPU | Ecosystem/tooling not yet supported | Accuracy vs slightly newer tooling |
| GGUF | CPU/edge/consumer GPU deployment | Max datacenter GPU throughput needed | Portability vs peak GPU speed |
| bitsandbytes/NF4 | QLoRA fine-tuning | Pure inference-serving throughput | Training-friendly vs not inference-optimized |
| W4A16 | Memory-bound, small-batch serving | Large-batch compute-bound serving | Bandwidth savings |
| W8A8/SmoothQuant | Compute-bound, large-batch serving | Small-batch memory-bound serving | Throughput via INT8 compute |

---

## 7. Mixture of Experts (MoE) & Routing

- **Basic MoE (Switch Transformer/Mixtral style)**: replace dense FFN with N experts; a router selects top-k experts per token; only selected experts computed.
  - Pros: decouples parameter count from FLOPs — huge total capacity at fixed compute-per-token; Mixtral 8x7B matches/beats larger dense models at lower inference FLOPs.
  - Cons: total memory footprint (all experts must be resident) is much higher than active-compute suggests; routing instability, load imbalance across experts.
  - Pick over dense FFN when: want more model capacity without proportional inference compute increase, and can afford the memory to hold all experts.

- **Softmax Router (standard top-k)**: router computes softmax over experts, picks top-k highest-scoring.
  - Pros: simple, differentiable, standard baseline.
  - Cons: can collapse to always routing to a few "popular" experts without an auxiliary balancing loss.

- **Load-Balancing Auxiliary Loss**: adds a penalty term encouraging uniform token distribution across experts (importance × load product, summed and scaled).
  - Pros: prevents expert collapse and under-utilization; standard in nearly all production MoE.
  - Cons: an extra loss term to tune (weight coefficient); imperfect — some imbalance persists.

- **Expert Choice Routing (Google, 2022)**: inverts the assignment — instead of tokens choosing experts, experts choose their top tokens from the batch.
  - Pros: guarantees perfect load balance by construction (no auxiliary loss needed); no token dropping.
  - Cons: batch-dependent (a token's routing depends on what else is in the batch) — complicates causal/streaming inference; less common in production than token-choice + aux loss.
  - Pick over softmax+aux-loss when: training-time batch composition is controllable and perfect balance is worth the batch-dependency complexity.

- **DeepSeek fine-grained shared + routed experts**: split experts into always-active "shared" experts (capture common knowledge) plus many small "routed" experts (capture specialized knowledge), increasing total expert count while keeping per-token compute fixed.
  - Pros: shared experts reduce redundancy across routed experts (each routed expert doesn't need to relearn common patterns); finer granularity improves specialization.
  - Cons: added architectural complexity; more hyperparameters (number of shared vs routed experts).

- **Capacity Factor / Token Dropping**: each expert has a fixed processing capacity per batch; tokens routed beyond capacity are dropped (skip that expert, sometimes skip the FFN entirely).
  - Pros: bounds compute/memory per expert deterministically, enabling static compilation/batching.
  - Cons: dropped tokens get degraded representations; capacity factor is a tunable tradeoff between drop rate and wasted compute (excess capacity).

- **Expert Parallelism**: shard experts across devices, requiring all-to-all communication to route tokens to the device holding their assigned expert.
  - Pros: enables total parameter count far beyond single-device memory.
  - Cons: all-to-all communication is a major bottleneck, especially at scale/low-bandwidth interconnect.

**Comparison table**

| Method | Best for | Avoid when | Key tradeoff |
|:---|:---|:---|:---|
| Token-choice + aux loss | Standard production default (Mixtral) | Need guaranteed perfect balance | Simplicity vs imperfect balance |
| Expert Choice | Training-time perfect balance | Streaming/causal inference constraints | Perfect balance vs batch-dependency |
| Shared+routed (DeepSeek) | Maximizing specialization at fixed compute | Simpler architecture preferred | Specialization vs complexity |
| Capacity factor/dropping | Deterministic compute bounds | Cannot tolerate any dropped tokens | Predictability vs quality loss on drops |

Failure modes: expert collapse (router always picks same few experts), token dropping (quality degradation under high load), routing instability (router logits oscillate during training), expert redundancy (multiple experts learn near-identical functions), all-to-all communication bottleneck at scale.

---

## 8. Tokenization (BPE vs WordPiece vs SentencePiece vs Tiktoken)

- **BPE (Byte-Pair Encoding)**: iteratively merge the most frequent adjacent symbol pair into a new token, starting from characters/bytes.
  - Pros: simple, greedy, data-driven; handles arbitrary vocabulary via merges; the foundational modern subword method.
  - Cons: purely frequency-based merges can be suboptimal vs likelihood-based methods; requires pre-tokenization (splitting on whitespace) in the original formulation.
  - Pick over WordPiece when: want the simplest, most widely supported baseline (GPT family).

- **Byte-level BPE (GPT-2+)**: operate on raw UTF-8 bytes (256 base vocab) instead of unicode characters, so any byte sequence is representable with no unknown-token fallback.
  - Pros: no OOV/unknown tokens ever, handles any language/emoji/binary robustly.
  - Cons: rare scripts fragment into more tokens (lower fertility), inflating sequence length for those languages.

- **WordPiece (BERT)**: like BPE, but merges are chosen by maximizing likelihood/PMI-based score rather than raw frequency; uses `##` prefix for word-continuation subtokens.
  - Pros: likelihood-based merge criterion can yield more linguistically coherent subwords than raw BPE frequency.
  - Cons: PMI-based merge scoring is more complex to implement than raw BPE; largely tied to BERT-family models now.

- **SentencePiece (T5, LLaMA, Mistral)**: treats input as a raw character/byte stream with no pre-tokenization (whitespace becomes an explicit `▁` marker token), supports both BPE and Unigram LM algorithms internally.
  - Pros: language-agnostic (no assumption of whitespace-delimited words — critical for CJK); fully reversible/lossless detokenization; the modern default for multilingual/LLM tokenizers.
  - Cons: Unigram LM variant is more complex than plain BPE; `▁` marker convention needs understanding when debugging token boundaries.
  - Pick over BPE/WordPiece when: multilingual support (especially non-whitespace-delimited languages) matters.

- **Tiktoken (OpenAI)**: fast, Rust-based BPE implementation with fixed encodings (r50k_base, p50k_base, cl100k_base for GPT-3.5/4, o200k_base for GPT-4o).
  - Pros: very fast encode/decode; large, well-tuned vocab (100K-200K) reduces fertility for code and multilingual text vs older 50K vocabs.
  - Cons: proprietary/fixed vocab per model family — can't easily retrain/modify; still standard BPE under the hood, not a new algorithm.

**Comparison table**

| Method | Best for | Avoid when | Key tradeoff |
|:---|:---|:---|:---|
| BPE (byte-level) | Simple, robust, no-OOV baseline | Need max multilingual fertility efficiency | Simplicity vs suboptimal merges |
| WordPiece | BERT-family compatibility | New model without BERT lineage | Legacy fit vs added complexity |
| SentencePiece | Multilingual/CJK, LLaMA-style models | Simplicity preferred over multilingual robustness | Language-agnostic vs implementation complexity |
| Tiktoken | OpenAI-ecosystem speed + large vocab | Need to train a custom vocab | Speed/vocab size vs fixed/proprietary |

Vocab sizing tradeoff: larger vocab → lower fertility (fewer tokens per word) and shorter sequences, but larger softmax output layer and embedding table (memory + compute at output layer scale with vocab size). Typical sizes: 32K (LLaMA-1/2), 100K-128K (LLaMA-3, GPT-4), 200K (GPT-4o). Artifacts to know: leading-whitespace-sensitive tokens, capitalization fragmenting tokens differently, numbers fragmenting into inconsistent digit groups, code tokenization inefficiency without dedicated training data.

---

## 9. Inference Optimization (KV Cache, PagedAttention, FlashAttention, Batching, Serving Frameworks)

- **KV Cache**: cache computed K/V projections across generation steps so each new token only computes attention against cached history, not recomputing from scratch.
  - Pros: essential for any practical autoregressive serving — turns O(n²) recompute into O(n) incremental cost.
  - Cons: memory footprint grows linearly with sequence length × batch size × layers × heads; can dominate total GPU memory at long context/high concurrency.
  - Formula: `KV cache bytes = 2 × L × H_kv × d_head × seq_len × batch × bytes_per_element`.

- **PagedAttention (vLLM)**: manages the KV cache like OS virtual memory — fixed-size non-contiguous pages with a page table, instead of requiring one contiguous memory block per sequence.
  - Pros: near-zero memory fragmentation/waste (vs naive contiguous allocation which reserves worst-case max length per sequence); enables much higher batch concurrency at same memory.
  - Cons: added indirection (page table lookups) and engineering complexity vs naive contiguous KV cache.
  - Pick over naive KV cache when: serving many concurrent variable-length requests — this is essentially always true in production, hence vLLM's dominance.

- **FlashAttention**: computes attention via tiling + online softmax so the full attention matrix is never materialized in slow HBM, only kept in fast on-chip SRAM.
  - Pros: exact (not approximate) attention, O(n) memory instead of O(n²), significantly faster wall-clock due to reduced memory reads/writes (I/O-bound, not compute-bound problem).
  - Cons: implementation is hardware/kernel-specific (custom CUDA); FA-2/FA-3 further optimize but require newer GPU architectures for full benefit.
  - Pick over standard attention when: essentially always, for both training and inference, when hardware support exists.

- **Batching strategies**: **static batching** (fixed batch, wait for all sequences to finish — wastes compute on early-finishing sequences), **dynamic batching** (add new requests between batches at fixed intervals), **continuous/iteration-level batching** (vLLM/TGI style — evict finished sequences and admit new ones at every decode step, no waiting).
  - Pick continuous batching when: production serving with variable-length, asynchronously-arriving requests — this is now the standard (vLLM, TGI, TensorRT-LLM all use it).

**Serving framework decision table**

| Framework | Best for | Avoid when | Key tradeoff |
|:---|:---|:---|:---|
| vLLM | General GPU serving, PagedAttention, continuous batching | Need NVIDIA-specific max-optimization | Broad support vs peak per-hardware speed |
| TensorRT-LLM | Max throughput on NVIDIA hardware | Need portability across hardware/fast iteration | Peak speed vs NVIDIA lock-in, harder to iterate |
| llama.cpp | CPU/edge/consumer GPU, GGUF models | Datacenter GPU max-throughput serving | Portability vs raw throughput |

Debugging slow inference: diagnose prefill-bound (long-prompt-dominated, compute-bound) vs decode-bound (long-generation-dominated, memory-bandwidth-bound) — different bottlenecks need different fixes (prefill: batch/parallelize; decode: reduce KV cache reads via quantization/GQA/MLA/speculative decoding). The "Memory Wall": modern GPUs have far more compute (FLOPs) than memory bandwidth, so at low batch sizes GPUs sit compute-idle waiting on weight/KV reads — this is the root cause behind quantization, speculative decoding, and MQA/GQA/MLA all mattering.

---

## 10. Speculative Decoding (Standard vs Medusa vs Eagle vs Self-Speculative vs MTP)

- **Standard Speculative Decoding**: small draft model proposes K tokens autoregressively; large target model verifies all K in one forward pass via rejection sampling; lossless (output distribution exactly matches target-only sampling).
  - Pros: exact correctness guarantee; large speedups (2-3x) when draft model tracks target well; no retraining of target model required.
  - Cons: needs a separate compatible (same-vocab) draft model to deploy/maintain; speedup collapses at large batch sizes (compute-bound regime) or with high-temperature sampling.
  - Formula: `E[tokens per target call] = (1-α^(K+1))/(1-α)`, α = per-token acceptance rate.

- **Medusa**: attach K extra linear prediction heads to the target model's final hidden state, each predicting a different future position; tree attention verifies multiple candidate paths at once.
  - Pros: no separate draft model — just extra lightweight heads on the frozen base; simpler deployment than standard speculative decoding.
  - Cons: lower acceptance rate (0.65-0.75) than a trained draft model since heads see only current hidden state with no recurrence; requires fine-tuning the heads.

- **Eagle / Eagle-2**: a small single-transformer-layer draft model conditions on both the token embedding and the target model's actual hidden state to predict the next hidden state, sharing the target's LM head.
  - Pros: much better-calibrated drafts than Medusa (uses real target internal state); Eagle-2's dynamic draft tree adapts to confidence, reaching 3-4x speedup.
  - Cons: tighter coupling to target model internals (needs access to hidden states); must retrain per target model.

- **Self-Speculative Decoding**: use an early-exit (e.g., first L/2 layers) of the target model itself as the draft, verified by the full model.
  - Pros: zero extra memory (no separate model/heads); simplest architecture.
  - Cons: lower acceptance rate than dedicated draft models; best only for deterministic tasks (code, factual QA).

- **Multi-Token Prediction / DeepSeek MTP**: train auxiliary heads to predict multiple future positions as part of the base training objective itself, so speculative tokens come for free at inference.
  - Pros: zero inference-time cost for drafting (heads computed anyway); no separate model or extra forward passes.
  - Cons: must be baked in from pretraining — cannot retrofit onto an existing model cheaply.

**Comparison table**

| Method | Best for | Avoid when | Key tradeoff |
|:---|:---|:---|:---|
| Standard (draft model) | High acceptance with a strong same-family small model | No suitable small draft model exists | Best speedup, needs a maintained 2nd model |
| Medusa | No separate model, moderate speedup | Need max acceptance rate | Simplicity vs lower acceptance |
| Eagle/Eagle-2 | Best speedup without full 2nd model | Can't access target hidden states (closed API) | Best quality drafts, tighter coupling |
| Self-speculative | Zero extra memory | Creative/high-entropy generation | No memory cost, lower acceptance |
| MTP (DeepSeek) | Baked into a new model from scratch | Retrofitting an existing deployed model | Free at inference, requires pretraining investment |

Rule of thumb: speculative decoding helps most at small batch size (1-4, memory-bound), long generations, and high-predictability tasks (code: 0.85-0.92 acceptance, 2.5-4x speedup); hurts at large batch (32+, compute-bound) and short/creative generations (creative writing: 0.60-0.75 acceptance, 1.5-2x).

---

## 11. Model Merging (Averaging vs SLERP vs Task Arithmetic vs TIES vs DARE vs Model Soup vs Frankenmerging)

- **Simple Weight Averaging**: `θ_merged = Σ αᵢθᵢ`. Works only if models share the same pretrained base and lie in the same loss basin (linear mode connectivity).
  - Pros: trivial to implement; cheap.
  - Cons: fails outright across different base checkpoints; can fail even with the same base if task directions conflict.
  - Pick over SLERP when: merging many (3+) models where pairwise SLERP ordering issues would complicate things.

- **SLERP**: spherical interpolation along the great-circle arc between two weight vectors, preserving magnitude (unlike linear averaging, which shrinks magnitude at the midpoint).
  - Pros: avoids the magnitude-shrinkage problem of linear averaging (up to 29% magnitude loss at midpoint for near-orthogonal vectors).
  - Cons: only defined for exactly two models; order-dependent when chained for 3+.
  - Formula: `SLERP(θ₀,θ₁,t) = sin((1-t)Ω)/sin(Ω)·θ₀ + sin(tΩ)/sin(Ω)·θ₁`, `Ω=arccos(θ₀·θ₁/(||θ₀||·||θ₁||))`.
  - Pick over linear averaging when: merging exactly 2 models and magnitude preservation matters.

- **Task Arithmetic**: compute task vectors `τ = θ_finetuned - θ_pretrained`, then add/subtract/scale them independently (`θ_new = θ_base + λ·τ`) to compose or remove capabilities.
  - Pros: enables capability composition (add math + code) and even capability removal (subtract a task vector to unlearn behavior); analogous to word-vector arithmetic.
  - Cons: task vectors from very different domains can be near-orthogonal (adding noise, not capability) or have sign conflicts causing cancellation.
  - Pick over plain averaging when: want to compose/remove specific capabilities rather than blend whole models.

- **TIES Merging**: Trim (zero small-magnitude deltas), Elect sign (majority vote per parameter across task vectors), Discard (zero out deltas disagreeing with elected sign), then average survivors.
  - Pros: directly resolves the sign-conflict interference problem in naive task arithmetic when merging 3+ task vectors.
  - Cons: requires choosing a trim threshold (too aggressive discards real signal, too lenient keeps noise); majority vote can override a genuinely useful minority contribution.
  - Pick over plain task arithmetic when: merging 3+ task vectors where sign conflicts are expected.

- **DARE**: randomly drop a large fraction (60-90%) of delta parameters per task vector, rescaling survivors to preserve expected magnitude (dropout applied to weight deltas). Used as a preprocessing step before averaging/TIES/task-arithmetic.
  - Pros: reduces inter-model interference, especially as the number of merged models grows.
  - Cons: stochastic (non-reproducible without fixed seed); very high drop rates can lose genuine task signal.

- **Model Soup**: average multiple fine-tuning runs from the *same* base/hyperparameter-search sweep. Uniform soup averages all; greedy soup adds candidates one at a time only if validation performance improves.
  - Pros: often beats any single checkpoint, improving both in-distribution and OOD robustness; cheap (no retraining).
  - Cons: requires all runs share the same pretrained init; greedy variant needs a held-out validation set; adds nothing if all runs converged to nearly identical weights.

- **Frankenmerging**: splice layers from different models together (e.g., early layers from model A, late layers from model B) rather than blending weights.
  - Pros: sidesteps weight-interference entirely for intact spliced layers; used to build larger models (e.g. 120B from two 70Bs via layer duplication).
  - Cons: no guarantee layer interfaces are compatible even with identical architectures — perplexity often spikes at splice boundaries; mostly trial-and-error.

**Comparison table**

| Method | Best for | Avoid when | Key tradeoff |
|:---|:---|:---|:---|
| Weight averaging | Same-basin models, quick baseline | Different base checkpoints | Simplicity vs fragility |
| SLERP | 2-model merge, magnitude preservation | 3+ model merges | Preserves scale, only pairwise |
| Task arithmetic | Composing/removing specific capabilities | Very divergent task domains | Compositionality vs orthogonality risk |
| TIES | 3+ task vectors with sign conflicts | Simplicity preferred, only 1-2 models | Resolves conflicts, adds threshold tuning |
| DARE | Reducing interference at scale (many models) | Need reproducibility | Less interference, stochastic |
| Model Soup | Same-sweep hyperparameter search checkpoints | Different base inits | Free accuracy/robustness gain, needs shared init |
| Frankenmerging | Building novel larger/hybrid models | Predictable, low-risk results needed | Flexibility vs high failure rate |

Prerequisites for successful merging: same architecture (required), same vocab/tokenizer (required), same pretrained base (strongly recommended), similar task domains (helpful). MergeKit (Arcee AI) is the standard tool — YAML-declarative, supports `linear`, `slerp`, `task_arithmetic`, `ties`, `dare_ties`, `dare_linear`, `passthrough` (Frankenmerging).

---

## 12. RAG (Retrieval-Augmented Generation)

- **Core idea**: ground generation in retrieved documents rather than relying on frozen parametric memory — makes knowledge updatable without retraining and answers source-attributable.
  - Pick RAG over fine-tuning when: facts change frequently, need source attribution, knowledge base is large, or need to add knowledge post-deployment. Pick fine-tuning when: need to change behavior/tone/format, teach reasoning patterns, or latency is critical (RAG adds 50-300ms retrieval overhead).

- **Chunking strategies**: fixed-size (simple, splits mid-sentence), sentence splitting (coherent but short), recursive character (general default, respects paragraph hierarchy), semantic chunking (highest quality, expensive), structure-aware (markdown/PDF headers).
  - Tradeoff: smaller chunks → precise retrieval, less context; larger chunks → more context, diluted retrieval score. Typical: 512-1024 tokens, 50-100 token overlap.

- **Dense retrieval** (embedding cosine similarity): handles paraphrase/semantics, misses exact-match (proper nouns, IDs).
- **Sparse retrieval (BM25)**: keyword-based, strong for exact-match, fails at paraphrase.
- **Hybrid (Reciprocal Rank Fusion)**: `RRF(d) = Σ 1/(k+r(d))` combines dense+sparse rank lists without needing calibrated scores — best of both, adds a fusion step.
  - Pick hybrid over dense-only when: queries include technical IDs/proper nouns alongside semantic questions — almost always the safer default in production.

- **ANN indexes**: **HNSW** (multi-layer navigable graph, O(log n) query, tunable via `ef`/`ef_construction`) vs **IVF/IVFPQ** (Voronoi-cell clustering, compressed storage via product quantization at cost of recall).
  - Pick HNSW when: recall/speed tradeoff needs fine control and memory is available. Pick IVFPQ when: memory-constrained at billion-scale.

- **Query optimization**: **Multi-query expansion** (generate reformulations, retrieve for each, dedupe — higher recall, more retrieval calls), **HyDE** (embed a hypothetical LLM-generated answer instead of the raw query — better when query is short/vague, but a wrong hypothetical answer misdirects retrieval), **Step-back prompting** (generalize the query before retrieving).

- **Reranking**: two-stage retrieval — fast ANN retrieves top-20 (bi-encoder), slow cross-encoder reranks to top-5 (joint query+doc processing, much more accurate but far slower per-candidate).
  - Pick reranking when: retrieval precision matters more than latency budget.

- **Self-RAG**: model learns (via fine-tuning with special tokens) to decide when to retrieve and self-assess relevance/support — avoids retrieving when unnecessary, but requires fine-tuning.
- **CRAG (Corrective RAG)**: evaluates retrieval quality, falls back to web search if poor — prevents confident hallucination from bad context, but adds latency and an imperfect quality evaluator.
- **GraphRAG**: builds a knowledge graph for multi-hop reasoning across entities — outperforms flat RAG on complex multi-hop questions, overkill for simple lookups.

**Comparison table**

| Technique | Best for | Avoid when | Key tradeoff |
|:---|:---|:---|:---|
| Dense retrieval | Semantic/paraphrase queries | Exact-match technical terms | Semantics vs exact match |
| BM25 | Exact-match, proper nouns/IDs | Paraphrased natural queries | Speed/precision vs no semantics |
| Hybrid (RRF) | Production default | Ultra-low-latency budget | Best recall, added fusion step |
| HyDE | Short/vague queries | Domain where LLM might hallucinate answer | Better recall vs misdirection risk |
| Reranking | High-precision top-k | Tight latency budget | Precision vs latency |
| Self-RAG | Avoiding unnecessary retrieval | No fine-tuning budget | Adaptive retrieval, needs training |
| CRAG | Guarding against bad retrieval | Latency-sensitive | Robustness vs added latency |
| GraphRAG | Multi-hop entity reasoning | Simple single-doc lookup | Deep reasoning vs pipeline cost |

RAGAS eval metrics: Faithfulness (is the answer grounded in context?), Answer Relevancy, Context Precision (are retrieved chunks relevant?), Context Recall (was necessary info retrieved?). Debug order for wrong answers: (1) missing/stale docs, (2) poor chunking, (3) irrelevant retrieval, (4) prompt not enforcing groundedness.

---

## 13. Agentic Workflows

- **Chain-of-Thought (CoT)**: model writes intermediate reasoning steps before the final answer, becoming part of its own context.
  - Pros: catches errors that a single forward pass would propagate silently; cheap (no external calls).
  - Cons: can produce confidently wrong but structurally coherent reasoning — reduces but does not eliminate multi-step error.
  - Pick over ReAct when: the task is pure reasoning with no need for external facts/tools.

- **ReAct (Reasoning + Acting)**: interleave Thought → Action (tool call) → Observation cycles, grounding reasoning in real tool outputs rather than parametric memory.
  - Pros: can access live/external facts CoT cannot; grounds reasoning in verified observations.
  - Cons: can hallucinate tool calls/arguments, loop indefinitely without a stop condition, consume context with long observations; every call adds latency/cost.
  - Pick over CoT when: task requires external/live information or computation.

- **Reflection/Self-Critique**: generate an answer, critique it, then revise using the critique as context.
  - Pros: critiquing is an easier task than generating perfectly the first time; catches some errors.
  - Cons: 2-3x token cost; model can rubber-stamp its own subtle errors; most reliable for code (traceable), least for subtle factual errors.

- **Tree of Thought (ToT)**: best-first search over multiple candidate reasoning branches, scoring and pruning at each step instead of committing to one linear path.
  - Pros: recovers from early bad choices; good for planning/combinatorial problems with multiple viable paths.
  - Cons: cost scales exponentially with branching factor/depth; the evaluator scoring branches can itself be unreliable.
  - Pick over Reflection when: the problem has genuinely multiple plausible solution paths, not just one path needing correction.

- **Tool Use / Function Calling**: give the model callable functions with typed schemas; host executes calls and returns results as context.
  - Failure modes: hallucinated tool names/args (mitigate with schema validation), wrong argument values, cascade errors from bad tool output, infinite loops (mitigate with hard max-iteration limits).

- **Multi-Agent Patterns**: **Orchestrator+subagents** (central planner dispatches to specialists), **Supervisor/worker** (state-graph routing, LangGraph-style), **Debate/critic** (two agents argue, third judges — reduces confirmation bias).
  - Pros: handles tasks exceeding a single context window/capability set.
  - Cons: multiplies every single-agent failure mode; 2-4x more LLM calls/latency; brittle inter-agent message formats; hard failure attribution.
  - Pick multi-agent over single-agent when: task genuinely spans distinct capability domains (research + code + writing) that don't fit one context well.

- **Routing**: a cheap classifier dispatches each query to the appropriate handler (RAG/code/search/direct) rather than sending everything through the most expensive path.
  - Pros: saves cost/latency on the majority of queries not needing the expensive path.
  - Cons: misrouting is invisible in the output (e.g., a query needing retrieval silently answered from stale parametric memory).

**Comparison table**

| Pattern | Best for | Avoid when | Key tradeoff |
|:---|:---|:---|:---|
| CoT | Pure reasoning tasks | Needs external/live facts | Cheap, no grounding |
| ReAct | Tasks needing tools/live data | Simple reasoning-only tasks | Grounded but costly/loop-prone |
| Reflection | Code and traceable-error tasks | Subtle factual domains | Cheap improvement, unreliable on subtle errors |
| Tree of Thought | Multi-path planning problems | Tight compute budget | Best recovery, exponential cost |
| Multi-agent | Genuinely multi-domain tasks | Single-domain tasks | Capability breadth vs cost/fragility multiplier |
| Routing | Mixed-difficulty query streams | Uniformly hard/simple queries | Cost savings vs silent misroute risk |

Guardrails: max-step limits, per-tool timeouts, output schema validation, content classifiers, human-in-the-loop for irreversible actions, sandboxed code execution. Observability: full trace of every LLM/tool call (inputs, outputs, latency, cost) — without it, multi-step failures are undebuggable.

---

## 14. Multimodal Fusion Techniques

- **ViT (Vision Transformer)**: split image into patches, flatten+project each to embedding dim, process as a transformer sequence (optionally with a CLS token).
  - Pros: unifies image processing into the same sequence format as text tokens.
  - Cons: CLS embedding loses spatial info; positional encoding degrades at resolutions other than training resolution.

- **CLIP**: jointly train image and text encoders with contrastive (InfoNCE) loss on matched pairs, aligning both into a shared semantic space.
  - Pros: enables zero-shot classification (compare image embedding to class-name text embeddings); strong general-purpose semantic alignment.
  - Cons: global embedding loses spatial/positional detail — poor for counting, localization, reading text at a location.
  - DINOv2 alternative: self-supervised, more spatially-aligned features — better for OCR/localization/document tasks inside a VLM than CLIP.

**Fusion architectures (how vision enters the LLM) — progression from cheap/shallow to deep/expensive:**

- **Linear Projection (LLaVA-1)**: project all ViT patch embeddings linearly (or via small MLP in LLaVA-1.5) into LLM embedding space, concatenate with text tokens; only the projector is trained, both ViT and LLM frozen.
  - Pros: cheapest to train (single small module); no architecture change to LLM.
  - Cons: one token per image patch — large context cost (256-1369+ tokens per image).
  - Pick over Q-Former when: training budget is minimal and context budget can absorb raw patch tokens.

- **Q-Former (BLIP-2)**: K learnable query tokens cross-attend to image features, compressing N patch embeddings down to K (~32) fixed tokens fed to the LLM.
  - Pros: ~8x reduction in context cost vs raw patches.
  - Cons: Q-Former itself needs pretraining (matching, contrastive, generation objectives) before use; lossy compression loses fine spatial detail.
  - Pick over linear projection when: context budget is tight (e.g., document/multi-image tasks) and the extra Q-Former pretraining cost is acceptable.

- **Perceiver Resampler (Flamingo)**: similar cross-attention compression to Q-Former, but interleaved with cross-attention gating layers throughout the LLM (not just at input) to control image influence per-layer.
  - Pros: finer-grained control of image influence throughout generation.
  - Cons: requires architectural modification to the LLM (new cross-attention modules) — cannot bolt onto an arbitrary frozen LLM.

- **Early/Native Fusion (Gemini, Llama 4, GPT-4o)**: image patches and text tokens enter a single unified token stream processed by the same transformer from layer 1 — no separate encoder/bridge module.
  - Pros: deepest possible cross-modal integration; the transformer itself learns to fuse modalities.
  - Cons: requires training from scratch on multimodal data at massive scale — cannot reuse a pretrained text-only backbone; research-frontier cost level.
  - Pick over late-fusion (linear/Q-Former/Perceiver) when: building a frontier-scale model from scratch with the compute budget to match; not practical for constrained teams.

**Comparison table**

| Fusion method | Best for | Avoid when | Key tradeoff |
|:---|:---|:---|:---|
| Linear projection | Cheap/fast bridging of frozen ViT+LLM | Tight context budget, many images | Cheapest, highest token cost |
| Q-Former | Context-constrained multi-image/document tasks | Minimal training budget | 8x token compression, added pretraining |
| Perceiver Resampler | Fine per-layer control of image influence | Cannot modify LLM architecture | Deeper control, requires LLM modification |
| Early/native fusion | Frontier-scale models built from scratch | Limited compute, need to reuse existing LLM | Deepest integration, massive training cost |

**Audio**: Whisper (log-mel spectrogram → ViT-style encoder → autoregressive text decoder; produces transcript only, discards tone/emotion/prosody) vs neural audio codecs (EnCodec/SoundStream — Residual Vector Quantization discretizes audio into codebook indices an LLM can generate autoregressively, enabling audio *generation* not just transcription; lossy, codes less interpretable than text tokens).

**Video**: Sparse frame sampling (cheap, misses fast motion/temporal dynamics) vs Factorized spatiotemporal attention (TimeSformer/ViViT — separate spatial/temporal attention, cheaper than full joint attention but approximates true joint interactions) vs Spacetime patches (Sora/DiT — full 3D patches modeled jointly, best temporal coherence, most compute).

Common multimodal failure modes: hallucinated objects (model "fills in" plausible content), spatial confusion (weak positional encoding for vision — mitigated by M-RoPE), counting errors (attention doesn't count explicitly), illegible fine text at low resolution, lost-in-the-middle across multiple images.

---

## 15. Training Stability & Systems

- **Loss spikes**: sudden loss increases mid-training, often from bad data batches or numerical instability — mitigated by gradient clipping, data curation, and restart-from-checkpoint-with-skip strategies.
- **Gradient clipping**: cap gradient norm to prevent single bad batches from destabilizing optimization — nearly universal in LLM training.
- **LR warmup + cosine decay**: linear warmup avoids early-training instability from large updates on randomly-initialized weights; cosine decay smoothly reduces LR for late-training convergence.
- **FP16 vs BF16**: FP16 has higher precision but a narrow exponent range (prone to overflow/underflow, needs loss scaling); BF16 has FP32's exponent range at half the precision (numerically safer, no loss scaling needed) — BF16 is now the standard for LLM pretraining.
- **ZeRO stages 1-3**: ZeRO-1 shards optimizer states across devices, ZeRO-2 additionally shards gradients, ZeRO-3 additionally shards parameters themselves — progressively more memory-efficient but more communication overhead.
  - Pick ZeRO-3 over ZeRO-1/2 when: model doesn't fit in memory even after basic sharding; accept the higher communication cost.
- **Checkpoint averaging**: average weights across the last N training checkpoints for a small, free quality boost (similar spirit to Model Soup but within one training run).
- **RLHF-specific failures**: reward hacking (Goodhart's Law — optimizing proxy diverges from true intent), KL explosion (policy drifts too far from reference, needs stronger KL penalty), DPO mode collapse (addressed by IPO's regularized objective).

---

## Quick-Reference: Pick-This-Over-That Index

| Decision | Pick A when... | Pick B when... |
|:---|:---|:---|
| GQA vs MLA | Standard production, simpler eng | Need extreme KV compression, long context |
| DPO vs RLHF | Static prefs, limited eng budget | Need online exploration, reusable reward model |
| LoRA vs Full FT | Task close to base distribution, limited compute | Large distribution shift, quality is paramount |
| QLoRA vs LoRA | GPU memory is binding constraint | Training speed matters more than memory |
| AWQ vs GPTQ | Need best 4-bit accuracy | Need broadest ecosystem/checkpoint support |
| GGUF vs GPTQ/AWQ | CPU/edge/consumer deployment | Datacenter GPU serving |
| YaRN vs NTK-aware | Large extension ratio (4K→128K+) | Moderate extension, simpler tuning |
| Standard spec-decoding vs Eagle | Only need a same-vocab small model | Want best possible speedup, can access hidden states |
| TIES vs plain task arithmetic | 3+ task vectors, sign conflicts likely | Only 1-2 vectors, low conflict risk |
| Hybrid RAG vs dense-only | Queries include exact IDs/proper nouns | Purely semantic/paraphrase queries |
| Multi-agent vs single-agent | Task spans genuinely distinct capability domains | Single-domain task fits one context |
| Q-Former vs linear projection | Context budget is tight | Training budget is minimal |
