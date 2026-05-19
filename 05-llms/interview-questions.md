# LLM Interview Questions — First-Principles Structure

Each question is structured as:
- **What the interviewer is testing**
- **The reasoning structure a first-principles thinker uses**
- **Common traps**

---

## Architecture & Theory

---

### 1. "Explain the KV Cache and why it's critical for real-time LLM applications."

**What the interviewer is testing**:
Whether you understand *why* inference is expensive, not just *that* it is cached. They want to see that you can trace the computational bottleneck to its source.

**First-principles reasoning structure**:

Start from the problem: autoregressive generation requires the model to attend over all previous tokens at every new generation step. For sequence length L, computing attention naively at each step costs O(L) per step, and O(L²) total. At L=4096, that is 16 million attention operations just to generate one token.

The insight: the Key and Value matrices for previous tokens are determined by those tokens' embeddings — they do not change as new tokens are added. Only the new token's Query changes. Therefore, cache the K and V matrices for all previous tokens after first computation, and only compute Q fresh for each new token.

The consequence: generation cost per token drops from O(L) to O(1) relative to sequence history. This is what makes interactive latency possible.

The cost: KV cache occupies GPU VRAM proportional to: `sequence_length × batch_size × num_layers × 2 (K and V) × d_model`. For a 70B parameter model with a 100k context window and large batch, KV cache alone can require hundreds of gigabytes of VRAM.

**Common traps**:
- Saying "KV Cache speeds things up" without explaining *what* it avoids recomputing and *why* that computation was redundant.
- Forgetting the memory cost. The interviewer may follow up: "What happens when the KV cache fills up?" The answer: eviction strategies (sliding window), or the context window limit is a hard constraint imposed by available VRAM, not just model design.
- Conflating the KV cache (inference optimization) with model checkpointing (training) or gradient caching.

---

### 2. "Why do we divide by √d_k in the scaled dot-product attention formula?"

**What the interviewer is testing**:
Whether you understand numerical stability in neural networks, not just the formula. A candidate who can derive *why* the scaling is necessary understands the training dynamics of attention.

**First-principles reasoning structure**:

Start from the operation: `QK^T` is a dot product between two vectors of dimension d_k. If each element of Q and K is drawn from a distribution with mean 0 and variance 1, the dot product of two d_k-dimensional vectors has variance d_k. As d_k grows (e.g., d_k = 64, 128, 512), the magnitude of these dot products grows proportionally.

The problem: large dot products are passed to the Softmax function. The Softmax of very large values saturates — one logit dominates, all others become near-zero. The gradient of a saturated Softmax is also near-zero. During backpropagation, gradients vanish. The attention weights stop learning.

The fix: divide by √d_k before Softmax. This normalizes the variance of the dot products back to approximately 1, regardless of d_k. Gradients remain informative throughout training.

Why √d_k specifically: if each component of Q and K has variance 1, the dot product (sum of d_k products) has variance d_k. To reduce variance to 1, divide by √d_k. It is a variance normalization, not an arbitrary constant.

**Common traps**:
- Memorizing "prevents large values" without connecting it to gradient behavior during training.
- Saying the scaling "helps the model focus on fewer tokens" — this confuses the effect of Softmax saturation (attention collapse to one token) with the cause (large pre-softmax values).
- Missing the √ — some candidates say "divide by d_k" (which would over-correct).

---

### 3. "Compare LoRA and Prefix-Tuning as PEFT methods."

**What the interviewer is testing**:
Whether you understand the architectural distinction between adapter methods (LoRA) and soft-prompt methods (Prefix-Tuning), and can reason about their practical consequences.

**First-principles reasoning structure**:

Start from the shared problem: full fine-tuning of a 70B model is infeasible for most practitioners. Both methods reduce the number of trainable parameters. But they do this through different mechanisms with different trade-offs.

**LoRA**: inserts trainable low-rank matrices as additive updates to existing weight matrices.
- `W_effective = W_0 + BA` where B is (d × r) and A is (r × d), r << d.
- The frozen original weights remain unchanged. The adapter is additive.
- Effect on context window: zero. LoRA does not consume any input token positions.
- The adapter can be added and removed without changing the model's architecture.

**Prefix-Tuning**: prepends trainable "virtual token" embeddings to the Key and Value sequences of each attention layer.
- These virtual tokens are not real input tokens — they are learned parameters that act as soft conditioning.
- Effect on context window: the prefix tokens occupy positions at the start of every layer's K and V. Effectively reduces the usable input length by the prefix length.
- For a prefix of length 20 with a 2048 context model: 2028 tokens available for actual input.

**Key distinction**: LoRA adapts the *weights* of the model. Prefix-Tuning adapts the *activations* by conditioning the attention mechanism with learned context.

**When to prefer each**:
- LoRA: tasks requiring new capabilities or domain adaptation. No context window penalty.
- Prefix-Tuning: tasks where "style" or "persona" conditioning is the primary need. Small number of trainable parameters.

**Common traps**:
- Describing Prefix-Tuning as "prepending tokens to the input" — the prefix is injected at every layer's K and V, not just the input embedding layer.
- Claiming LoRA "changes the architecture" — it does not. The adapter weights are summed with the original weights at inference, resulting in a standard-shaped weight matrix.
- Missing the context window trade-off of Prefix-Tuning entirely.

---

## Training & Production

---

### 4. "How do you detect and mitigate catastrophic forgetting when fine-tuning an LLM?"

**What the interviewer is testing**:
Whether you understand the tension between specialization and generalization in neural networks, and have practical strategies for managing it.

**First-principles reasoning structure**:

Start from the mechanism: neural networks store "knowledge" as distributed patterns of weights. When you fine-tune on task-specific data, gradient updates push all weights toward minimizing loss on that task. If the fine-tuning data distribution differs significantly from pre-training, these updates can destructively overwrite the weight configurations that encoded pre-training knowledge.

**Detection**: run a general-purpose benchmark (MMLU, HellaSwag, or the task the base model was originally strong at) before and after fine-tuning. A significant drop in general benchmark score indicates forgetting. Monitor: if domain-specific score improves while general score degrades, forgetting has occurred.

**Mitigation strategies**:

*Prevention* (before training):
- Low learning rate (1e-5 to 3e-5): small gradient steps minimize overwriting of pre-trained representations.
- PEFT (LoRA): by freezing the original weights entirely and training only adapter parameters, the pre-trained knowledge is structurally preserved. Forgetting requires actually updating weights that encode prior knowledge — LoRA does not touch them.
- Experience replay: mix a small fraction (5-15%) of general pre-training data into the fine-tuning batch. The model continues to receive gradient signal to maintain general capabilities.

*Detection* (during training):
- Track a held-out "general capability" evaluation set. Stop training if the general score drops more than an acceptable threshold (e.g., 5 percentage points below baseline).

**Common traps**:
- Treating forgetting as a bug to be debugged rather than an inherent tension in neural learning.
- Claiming "use a larger model" — parameter count does not prevent forgetting; it may reduce its severity, but the mechanism is the same.
- Forgetting that PEFT is the most practically effective mitigation for fine-tuning at scale, not just regularization tricks.

---

### 5. "What is RLHF and why can't we just use Supervised Fine-Tuning (SFT) for alignment?"

**What the interviewer is testing**:
Whether you understand the fundamental limitation of SFT for preference learning, and can articulate why indirect optimization through a reward model adds value that direct supervision cannot.

**First-principles reasoning structure**:

Start from what SFT can and cannot encode:

SFT trains the model to imitate the distribution of human-labeled responses. For a given prompt, the model learns "produce output similar to the average of what labelers wrote." Problems:
1. Human labelers vary in quality. SFT averages over their distribution, including low-quality examples.
2. Complex human preferences (tone, safety, nuanced appropriateness) are difficult to express in a single "ideal response." What makes a response good or bad is often comparative, not absolute.
3. SFT cannot express "prefer A over B" — it can only express "produce A."

RLHF solves the comparison problem:
1. Generate multiple candidate responses to the same prompt.
2. Have humans *rank* them (A > B > C). This is easier and more reliable than writing the ideal response from scratch.
3. Train a Reward Model to predict these human rankings from (prompt, response) pairs.
4. Use PPO (Proximal Policy Optimization) to optimize the LLM to maximize Reward Model scores.

The reward model distills the human preference signal. The LLM is then optimized against this signal with reinforcement learning, which allows the model to explore response variations that the original SFT data did not contain.

DPO (Direct Preference Optimization) achieves similar results without a separate Reward Model by directly optimizing the probability ratio between preferred and rejected responses.

**Common traps**:
- Saying "SFT doesn't use human feedback" — SFT absolutely uses human-written responses, which are a form of human feedback. The distinction is *comparative* vs. *absolute* feedback.
- Confusing the Reward Model with the policy (the LLM being aligned). These are separate models.
- Not knowing the cost of RLHF: it requires three separate training runs (SFT, Reward Model, PPO) and is notoriously unstable. This is why DPO was developed.

---

### 6. "Describe the Chinchilla scaling laws in your own words."

**What the interviewer is testing**:
Whether you can extract the practical engineering implication from the research result, not just recite the paper's finding.

**First-principles reasoning structure**:

Start from the question the paper answered: "Given a fixed compute budget (measured in FLOPs), how should you split it between model size and training tokens?"

Prior practice (GPT-3 era): researchers assumed "bigger model = better." They trained very large models on relatively small amounts of data. GPT-3 (175B parameters) was trained on ~300B tokens.

The Chinchilla finding: the optimal allocation is approximately equal scaling between parameters and training tokens. For every parameter, you should train on roughly 20 tokens. GPT-3-scale compute would be better spent on a 70B-parameter model trained on 1.4T tokens rather than a 175B model trained on 300B tokens.

The practical implication: most pre-Chinchilla models were *overtrained in size and undertrained in data*. Chinchilla (70B parameters, 1.4T tokens) outperformed GPT-3 (175B, 300B tokens) at the same compute cost.

The follow-on implication: Llama 2 (7B, 2T tokens) outperforms much larger models on many benchmarks because it was trained on far more data than Chinchilla-optimal. Trading model size for inference efficiency: a smaller model trained on more data is faster at inference and easier to deploy than a larger model trained on less data with equivalent capability.

**Common traps**:
- Treating the "20 tokens per parameter" ratio as a hard law rather than an approximation from a specific compute range. It may not hold at extreme scales.
- Missing the inference cost implication: smaller models that are trained longer have better inference economics (lower latency, lower cost per query) while matching larger undertrained models on capability benchmarks.
- Confusing Chinchilla the model with Chinchilla the paper. The paper's contribution is the scaling law; the model demonstrated it.

---

## Problem-Solving

---

### 7. "An LLM is hallucinating a legal fact. How do you solve this without retraining?"

**What the interviewer is testing**:
Whether you understand that hallucination is a knowledge-gap problem, not a reasoning-failure problem, and that retrieval is the correct intervention for factual grounding.

**First-principles reasoning structure**:

Diagnose the root cause: the model generates a plausible-sounding legal fact because it has learned the *form* of legal reasoning from training data, but it does not have the specific fact in its weights (or it has conflicting facts from different sources that it averages). Hallucination is the model filling a knowledge gap with statistical plausibility.

Retraining is the wrong intervention for factual grounding:
- Training data has a cutoff.
- Laws change. A retrained model will hallucinate new amendments.
- You cannot teach every jurisdiction's legal code through fine-tuning without degrading general capability.

The correct intervention: RAG (Retrieval-Augmented Generation).

Step by step:
1. Maintain a verified, up-to-date database of the relevant legal texts.
2. At query time, embed the question and retrieve the most relevant sections using semantic search.
3. Inject the retrieved text into the prompt: "Using only the following text, answer the question: [retrieved text]"
4. Instruct the model explicitly: "Do not generate any legal claims not supported by the provided text. If the answer is not in the provided text, say so."
5. Optionally: return citations (which document, which section) so the output is auditable.

Additional safeguards: a confidence threshold — if no retrieved document exceeds the similarity threshold, return "insufficient information" rather than allowing the model to fill the gap.

**Common traps**:
- Suggesting prompt engineering alone ("tell the model to be accurate") — instructions do not fill knowledge gaps; they only modulate behavior within the model's existing knowledge.
- Treating RAG as a complete solution without addressing retrieval quality. If the retrieval returns the wrong statute, the model grounds its answer in the wrong text.
- Suggesting fine-tuning as the first intervention — fine-tuning on legal text would improve general legal reasoning but cannot guarantee accuracy on any specific fact, especially one that changes over time.

---

### 8. "Your model is too large to fit on a single 80GB A100 GPU. What are your options?"

**What the interviewer is testing**:
Whether you know the full toolbox for managing model memory at inference and training time, and understand the trade-offs between options.

**First-principles reasoning structure**:

Start from the problem: a 70B parameter model in fp16 requires approximately 140GB to load (2 bytes × 70B parameters). An A100 has 80GB. The model does not fit.

Enumerate the intervention points:

**1. Quantization** (reduce precision):
- 8-bit (int8): ~70GB. Fits on the A100 with minimal accuracy loss.
- 4-bit (int4/NF4, as in QLoRA): ~35GB. Fits easily, with small accuracy degradation on some tasks.
- Mechanism: store weights as lower-precision integers; dequantize to fp16 during computation.
- Trade-off: faster loading, smaller VRAM footprint, small accuracy cost. Inference latency may increase or decrease depending on implementation.

**2. Model Parallelism** (split across GPUs):
- **Tensor parallelism**: split individual weight matrices across GPUs. Each GPU holds a shard; each forward pass requires cross-GPU communication. Low latency when GPUs are on the same node.
- **Pipeline parallelism**: different layers on different GPUs. Layer 0-10 on GPU 1, layers 11-20 on GPU 2. Forward pass goes through GPUs in sequence. Higher latency due to pipeline bubbles, but works across nodes.
- Tools: DeepSpeed, Megatron-LM, Accelerate.

**3. CPU Offloading**:
- Load some layers to system RAM, stream to GPU for computation.
- Dramatically slower (PCIe bandwidth is ~32 GB/s vs. GPU-to-GPU at ~600 GB/s).
- Used for inference where latency is not critical. Not practical for interactive applications.

**4. LoRA (training only)**:
- If the goal is fine-tuning rather than inference: LoRA freezes the base model and trains only adapter weights. The base model can be quantized to 4-bit (QLoRA), fitting in ~35GB. Adapter parameters are small.

**5. Knowledge Distillation** (longer-term option):
- Train a smaller student model to replicate the larger model's outputs.
- Produces a permanently smaller model that runs on less hardware.
- Expensive upfront; pays off at scale.

**Decision framework**:
- Inference only: quantization first (easiest, minimal accuracy cost), then multi-GPU if needed.
- Fine-tuning on one GPU: QLoRA.
- Production multi-model serving: model parallelism with a serving framework (vLLM, TGI).

**Common traps**:
- Listing options without discussing trade-offs. The interviewer wants to know *when* you'd choose each.
- Confusing tensor parallelism and pipeline parallelism — they have fundamentally different communication patterns and latency profiles.
- Not knowing the practical numbers: a 70B model in fp16 is ~140GB. In int8 it's ~70GB. In 4-bit it's ~35GB. These numbers should be recalled without calculation.
- Forgetting that quantization affects activation memory and KV cache in addition to weight memory. Full memory footprint is not just the weights.
