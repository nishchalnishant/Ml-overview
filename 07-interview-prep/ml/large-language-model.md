# Large Language Models

---

## What This File Is For

Every topic is structured around the four questions that matter in an interview:
1. What the interviewer is actually testing
2. The reasoning structure — why first-principles thinkers approach it this way
3. The pattern in action — a worked example
4. Common traps — where people go wrong and why

---

## 1. What an LLM Is and Why Scale Matters

**What the interviewer is testing:** Whether you understand the pretraining objective and why it produces capable models — not just that "LLMs are big transformers." The specific competency is connecting the training objective to the emergent capabilities, not reciting architecture details.

**The reasoning structure:** An LLM is a neural network — almost always Transformer-based — trained to predict the next token in a sequence. That objective sounds narrow. It becomes powerful because language is a lossy compression of everything humans know and intend. To predict the next token accurately across diverse text, the model must learn grammar, world knowledge, reasoning patterns, stylistic conventions, factual associations, and causal relationships — not because any of those were the explicit target, but because they are all useful for the prediction task.

Scale amplifies this for two reasons. First, larger models have greater representational capacity to store more world knowledge and more complex reasoning patterns. Second, emergent capabilities — abilities that appear suddenly at certain parameter counts and do not appear to be simple extrapolations of smaller models' behavior — arise at scale. Multi-step reasoning, instruction following, and in-context learning are emergent in this sense. The mechanism for emergence is not fully understood; the empirical observation is well-documented.

**The pattern in action:** A model trained only on next-token prediction on internet text learns to write code, explain concepts, translate languages, and perform arithmetic — none of which were specified as training objectives. To predict `print("Hello World")` in context, the model must learn that Python strings use double quotes, that print is a function call, and that indentation is syntactically meaningful. These behaviors emerge from the prediction task because they are necessary to minimize prediction loss on code text.

**Common traps:**
- Treating scale as the only variable. Architecture choices, data quality, and training curriculum all interact with scale. A larger model trained on noisier data can underperform a smaller model trained on curated data — data quality matters as much as parameter count.
- Treating emergent capabilities as magical. They are well-defined empirically: performance on a task is near-random at small scale and jumps discontinuously at large scale. The explanation is likely that some tasks require many capabilities simultaneously; the model acquires each capability gradually, but only crosses the threshold for all of them at once at large scale.

---

## 2. Tokenization

**What the interviewer is testing:** Whether you understand tokenization as a design decision with downstream consequences for cost, context window use, and domain transfer — not just "text gets split into pieces."

**The reasoning structure:** Before processing, text is converted to a sequence of integer token IDs. The tokenizer defines what counts as a token and determines the vocabulary size.

The central tension is vocabulary size versus sequence length. A large vocabulary produces shorter sequences (common words get single tokens) but requires a large embedding table and handles rare words poorly. A small vocabulary produces longer sequences but generalizes better to novel words and reduces embedding memory.

**Byte-Pair Encoding (BPE)** navigates this tradeoff by iteratively merging the most frequent adjacent byte pairs in the training corpus, producing a vocabulary where common words are single tokens and rare words decompose into recognizable subword pieces. Starting from individual characters (or bytes), merging the most frequent pair at each step, and stopping at a target vocabulary size (typically 32k–100k) produces a compression scheme that handles any text without unknown tokens.

Why tokenization design decisions matter in practice:
- Token count drives API cost for commercial models
- Token count determines how much content fits in the context window
- Domain-specific text (code, medical, legal) may tokenize inefficiently under a general-purpose vocabulary, inflating sequence length and fragmenting semantic units
- Languages differ in tokenization efficiency — English-centric tokenizers produce long sequences for morphologically rich languages

**The pattern in action:** The identifier `getUserAuthTokenFromCacheOrRefresh` tokenizes to 9–12 subword fragments under a standard English-trained tokenizer, making it difficult for the model to treat it as a semantic unit. A tokenizer trained to include common code patterns handles it as 2–3 tokens, improving both efficiency and representation quality. The code-specific vocabulary reduces token count by ~40% on typical code, which means the same context window fits 40% more code.

**Common traps:**
- Treating tokenization as fixed infrastructure. Vocabulary extension, fine-tuned tokenizers, and domain-specific vocabularies are real design choices with meaningful downstream effects.
- Assuming all languages tokenize equally efficiently. Thai, Arabic, and CJK languages can require 3–5× more tokens per sentence than English under English-optimized tokenizers, disproportionately consuming context window and increasing cost for multilingual applications.

---

## 3. The Transformer Architecture

**What the interviewer is testing:** Whether you can explain what the architecture actually does — grounding it in the problems it was designed to solve — not just list its components.

**The reasoning structure:** Transformers were designed to solve two problems that plagued prior sequence models:

1. **The sequential training bottleneck:** RNNs process tokens one at a time, meaning you cannot parallelize across the sequence dimension during training. Transformers compute attention over all positions simultaneously, enabling full parallelization across the sequence dimension. This is why Transformers can scale to much larger models and datasets.

2. **The information bottleneck:** RNNs compress the entire history into a fixed-size hidden state. Long-range dependencies require signal to travel through many sequential updates, where it can decay or be overwritten. Attention connects any two positions directly, regardless of their distance, with no intermediate steps.

The core operations in a Transformer block:
1. **Multi-head self-attention:** each position queries all other positions and aggregates their values weighted by relevance
2. **Add and norm:** residual connection plus layer normalization
3. **Feed-forward network:** two linear layers with a nonlinearity (typically GeLU or SwiGLU), applied independently per position
4. **Add and norm:** residual connection plus layer normalization

The feed-forward network is often described as the "memory" of the Transformer — it stores factual associations learned during pretraining, while attention handles dynamic routing of information within context. Evidence: ablating attention produces models that cannot track dependencies; ablating FFN produces models that lose factual recall.

**The pattern in action:** A Transformer processes a 1,000-token document in one forward pass and produces attention patterns that directly link a pronoun at token 950 to its antecedent at token 12. An LSTM requires the information from token 12 to survive 938 sequential state updates — each of which can overwrite it — to reach token 950.

**Common traps:**
- Treating the Transformer architecture as monolithic. Decoder-only (GPT, LLaMA), encoder-only (BERT), and encoder-decoder (T5, Bart) are all Transformers with different attention masking schemes optimized for different tasks. The masking pattern determines what each position can attend to, which determines what the model is capable of.
- Conflating the architecture with the training objective. A Transformer is an architecture; next-token prediction is an objective; BERT's masked language modeling is a different objective on the same architecture. Both choices are independent.

---

## 4. Attention and QKV

**What the interviewer is testing:** Whether you understand the mechanism of attention at a level that lets you reason about its behavior and failure modes — not just recite the formula.

**The reasoning structure:** The Query-Key-Value framing gives each attention head a clean role:
- **Query:** what this position is looking for
- **Key:** what each position offers for matching
- **Value:** the information carried if the match is strong

Scaled dot-product attention:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

The $\sqrt{d_k}$ scaling prevents dot products from becoming large when embedding dimension is high, which would push the softmax into saturation where gradients near-vanish. If $q_i, k_i \sim \mathcal{N}(0,1)$, then $\text{Var}(q^Tk) = d_k$, so dividing by $\sqrt{d_k}$ normalizes the variance to 1 regardless of dimension.

Multi-head attention runs $h$ independent attention operations in projected subspaces and concatenates the results:
$$\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O$$
$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

Different heads specialize in different relationship types: one might track syntactic dependencies, another coreference, another positional proximity. The heads learn these specializations from data — no explicit supervision assigns roles.

**The pattern in action:** In a coreference resolution task, one attention head learns to strongly link pronouns to their antecedents across long distances; another tracks subject-verb agreement within clause boundaries. Visualizing attention weights for the pronoun "she" in a document reveals one head attending strongly to the antecedent noun phrase regardless of distance between them. Neither behavior was specified — it emerged as a learned strategy for minimizing prediction loss.

**Common traps:**
- Saying attention tells the model "which words are important" without specifying: important for what? Attention patterns are query-dependent — the same word is important for different queries in different ways. Describing attention as a static importance score misses this.
- Treating attention weights as explanations of model reasoning. Attention weights are intermediate computations, not causal explanations. A position can have high attention weight but contribute little to the final output if its values are small.

---

## 5. Positional Encoding

**What the interviewer is testing:** Whether you understand why position information must be injected, and what different approaches trade off — specifically their behavior at context lengths beyond training.

**The reasoning structure:** Self-attention is permutation-equivariant — if you shuffle the input tokens, the output of attention shuffles identically. Without positional information, "John hit Mary" and "Mary hit John" produce identical attention-weighted representations. Order is invisible to the attention operation.

Positional information must be injected either at the input (as embeddings added to token embeddings) or at the attention score computation.

**Absolute sinusoidal encoding** (original Transformer): deterministic position-specific vectors using sine and cosine at different frequencies. $PE_{(pos,2i)} = \sin(pos/10000^{2i/d})$. Does not generalize beyond the training context length because positions beyond that range were never seen during training.

**Learned absolute positional embeddings** (BERT, GPT-2): a lookup table of position-specific embeddings trained alongside the model. Same limitation on context length generalization.

**Relative positional encodings — RoPE** (LLaMA, GPT-NeoX): rotates query and key vectors before the dot product so that the attention score depends naturally on relative position. For positions $m$ and $n$, the dot product of the rotated vectors depends only on $(m-n)$. Generalizes better to contexts longer than those seen during training; enables effective context extension via position interpolation.

**ALiBi** (Bloom): adds a negative bias proportional to distance to attention logits. Simple, does not modify embeddings, and can extrapolate to longer contexts by applying larger biases to distant positions. Effective for long documents without positional embedding table changes.

**The pattern in action:** Models using RoPE can be fine-tuned to handle contexts significantly longer than their pretraining context window through position interpolation — scaling the position indices to fit within the original range. Models using absolute positional embeddings have no position representation for out-of-range indices and require architectural changes or full retraining to extend context length.

**Common traps:**
- Treating positional encoding as a solved problem. The choice meaningfully affects long-context generalization, which matters increasingly as context windows extend to 128k–1M tokens.
- Confusing what is encoded. Positional encoding tells the model where each token is, not what it means. It enables order sensitivity; it does not encode semantic relationships.

---

## 6. Pretraining vs Fine-Tuning vs Prompting vs RAG

**What the interviewer is testing:** Whether you can reason about which approach is right for a given problem based on what the problem actually is — not which technique sounds most advanced.

**The reasoning structure:** These are four different levers for adapting model behavior, each appropriate for different problem types.

**Pretraining:** train from scratch on a large general corpus. Appropriate only when you have a fundamentally different domain, script, or modality that existing pretrained models do not handle. Rarely the right choice for application development — the cost is prohibitive and existing pretrained models cover most use cases.

**Fine-tuning:** update model weights on a curated dataset for a specific task or domain. Changes the model's underlying behavior — its reasoning style, output format, response style, or domain-specific knowledge. Appropriate when the behavior you need is consistent enough to learn from examples and prompting alone is insufficient. PEFT methods (LoRA, adapter tuning) reduce compute cost significantly by only updating a small fraction of parameters while achieving comparable results to full fine-tuning on many tasks.

**Prompting (zero-shot, few-shot, chain-of-thought):** give the model instructions and examples in context. Changes behavior at runtime without modifying weights. Appropriate when the task can be described in language, the model already has the required knowledge, and you need flexibility across tasks. Cost: only the tokens.

**RAG (Retrieval-Augmented Generation):** fetch relevant documents from an external store and inject them into the prompt context. Appropriate when the problem is knowledge grounding or freshness — the model needs information it could not have seen during training, or information that changes frequently. Cost: retrieval infrastructure plus the tokens for injected context.

**Decision framework:**
- Is the problem that the model does not know something, or that knowledge changes? → RAG
- Is the problem that the model's behavior style, format, or domain fluency is wrong? → fine-tuning
- Is the problem solvable by better instructions or examples? → prompting
- Does the problem require the model to internalize something deeply into its parametric memory? → fine-tuning or pretraining

**The pattern in action:** A team wants to improve a model's answers to internal policy questions. They consider fine-tuning. The right question: is this a knowledge problem or a behavior problem? If the model gives correctly-formatted answers but has outdated or missing policy content, that is a knowledge problem — RAG will solve it faster, more safely, and with immediate access to current documents. If the model gives content-correct answers but in the wrong format or tone, that is a behavior problem — fine-tuning is appropriate. Distinguishing these prevents expensive misapplication of fine-tuning to a problem that does not require weight updates.

**Common traps:**
- Defaulting to fine-tuning for knowledge freshness. Fine-tuning bakes knowledge into weights at a fixed point in time; it does not update when facts change. RAG solves freshness natively.
- Believing prompting can substitute for fine-tuning when the behavior change required is subtle and consistent. Long system prompts are expensive, brittle, and cannot match the reliability of weight-level behavior changes for consistent stylistic or format requirements.

---

## 7. Context Window

**What the interviewer is testing:** Whether you understand context windows as a resource with costs and limits — not just a number to maximize.

**The reasoning structure:** The context window is the maximum number of tokens the model can attend to simultaneously. Everything present — system prompt, conversation history, retrieved documents, user input, the model's own output — competes for this space.

Larger context windows enable: longer document comprehension, more conversation history, richer retrieval injection, fewer out-of-context truncation errors.

Larger context windows also increase:
- Attention computation cost: standard self-attention scales as $O(n^2)$ in sequence length
- Latency: longer sequences take longer to process
- Cost: API pricing typically scales with total tokens processed

There is also a representational limit: beyond a certain distance, transformer models exhibit "lost in the middle" behavior — content in the middle of a very long context receives less attention than content at the beginning or end, even with unlimited context window capacity. The model is more reliable on information near the start or end of the context.

**The pattern in action:** A RAG system with a 32,000-token context window naively stuffs in the top-50 retrieved chunks and produces worse answers than one using a reranker to select the top-5 most relevant chunks. More context is not always better — coherent, focused context outperforms noisy, comprehensive context because the model has less irrelevant material to route around.

**Common traps:**
- Treating context window size as the primary evaluation criterion for model selection. The model's ability to reason over long contexts (tested empirically) often matters more than the raw capacity. A model with 128k context but poor mid-context retrieval is less useful than a model with 32k context that reliably uses all of it.
- Ignoring the quadratic compute cost of extending context. Doubling the context window quadruples attention computation. Efficient attention variants (Flash Attention, linear attention approximations) mitigate this but do not eliminate it.

---

## 8. Temperature and Sampling

**What the interviewer is testing:** Whether you understand that temperature controls the shape of the output distribution — not the quality of the model's knowledge — and can reason about when to use different values.

**The reasoning structure:** At each generation step, the model outputs a probability distribution over the vocabulary. Temperature $T$ scales the logits before the softmax:

$$P(w_i) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

At $T < 1$: the distribution sharpens around the most probable tokens — outputs are more deterministic and focused. At $T > 1$: the distribution flattens — the model samples more uniformly, producing more diverse outputs. At $T \to 0$: greedy decoding (always pick the most probable token). At $T \to \infty$: uniform random sampling over the vocabulary.

Temperature does not change what the model knows. It changes how it samples from its learned distribution. A model that does not know the answer will not know it better at low temperature — it will confidently confabulate a wrong answer rather than a varied wrong answer.

Other sampling strategies:
- **Top-k sampling:** sample only from the $k$ most probable tokens, zeroing the rest. Fixed cutoff regardless of distribution shape.
- **Top-p (nucleus) sampling:** sample from the smallest set of tokens whose cumulative probability exceeds $p$ — the set shrinks when the model is confident and grows when it is uncertain. Adapts to distribution shape at each step.
- **Beam search:** maintain $k$ hypotheses in parallel, selecting the highest-probability complete sequence. Deterministic; often too conservative for creative tasks.

**The pattern in action:** For a factual question-answering system, temperature 0.1–0.3 concentrates the output on the model's highest-confidence tokens, reducing hallucination rate. For a creative brainstorming assistant, temperature 0.7–1.0 produces more varied and unexpected suggestions. For code generation with a required exact format, temperature near 0 with deterministic sampling eliminates variance at the cost of diversity.

**Common traps:**
- Framing temperature tuning as "improving the model." Temperature is a sampling parameter. The model's calibration — whether its top-probability tokens correspond to correct answers — comes from pretraining and fine-tuning, not from sampling parameters.
- Using only temperature when top-p often gives better control. Temperature alone can produce either very focused or very random outputs; top-p adapts to the model's own uncertainty at each step and typically produces more coherent diverse outputs.

---

## 9. Causal Masking and Autoregressive Generation

**What the interviewer is testing:** Whether you understand why decoder-only models use causal masking, what it means for training parallelism, and what KV-cache optimization actually does.

**The reasoning structure:** During training, the model sees the entire sequence at once and is trained to predict each token from only the tokens preceding it. Causal masking enforces this by setting attention weights from any position to any later position to zero — this prevents the model from "seeing the answer" during training while enabling full parallelism across all sequence positions.

$$M_{ij} = \begin{cases} 0 & j \leq i \\ -\infty & j > i \end{cases}$$

Adding $M$ to attention logits before softmax means future positions receive zero weight.

During inference, generation is autoregressive: the model generates one token, appends it to the context, and generates the next token from the expanded context. This is inherently sequential — you cannot parallelize across generation steps because each step requires all previous outputs.

**KV-cache optimization:** during autoregressive generation, the key and value projections for previously generated tokens do not change between steps. They can be cached and reused rather than recomputed. Without KV-cache: step $n$ requires computing attention over $n$ tokens, so the total cost of generating $T$ tokens is $O(T^2)$. With KV-cache: each step adds only the new token's KV projections and runs attention using stored keys and values, reducing per-step cost to $O(n)$ incremental computation. The tradeoff is memory: caching scales linearly with context length, batch size, number of layers, and number of heads.

**The pattern in action:** A model generating a 500-token response without KV-caching recomputes attention over the full growing context at every step — step 499 attends over 500+ tokens even though only one new token was added. With KV-caching, step 499 adds one new KV pair and attends using the cached values for all previous tokens. For a batch of 32 requests generating 500 tokens each on a 70B model, KV-cache reduces inference FLOPs by 2–3 orders of magnitude.

**Common traps:**
- Conflating causal masking with the model not being able to see previous context. Causal masking prevents seeing future context. All previous tokens are fully attended to at every step.
- Ignoring KV-cache memory as a constraint. For long contexts and large batches, the KV-cache can exceed available GPU memory, requiring techniques like sliding window attention or KV-cache eviction to manage.

---

## 10. Instruction Tuning and RLHF

**What the interviewer is testing:** Whether you understand how pretrained base models are turned into useful assistants, and what each alignment technique actually optimizes for.

**The reasoning structure:** A pretrained base model is trained to predict text — it will continue whatever you give it. It is not trained to be helpful, to follow instructions, or to refuse harmful requests. Bridging this gap requires additional training stages.

**Supervised fine-tuning (SFT):** train on curated examples of instruction-response pairs. The model learns to respond to instructions in a helpful format. Necessary but often not sufficient — SFT models can be helpful but may not reliably respect human preferences or consistently refuse harmful requests.

**RLHF (Reinforcement Learning from Human Feedback):**
1. Humans compare pairs of model outputs and indicate which is better
2. A reward model is trained on these comparison pairs
3. The LLM is fine-tuned with RL (typically PPO) to maximize the reward model's score, subject to a KL divergence penalty against the SFT model to prevent the policy from drifting too far:

$$\text{Reward}_{\text{total}} = r_\theta(x, y) - \beta \cdot D_{KL}(\pi_\phi(y|x) \| \pi_{\text{SFT}}(y|x))$$

RLHF improves helpfulness and harmlessness. It also introduces failure modes: reward hacking (the model learns to game the reward model rather than genuinely satisfy the underlying preference) and capability degradation (too much RL fine-tuning can damage the pretraining-acquired capabilities). The KL penalty constrains this drift.

**DPO (Direct Preference Optimization):** directly optimizes the policy on preference data without a separate reward model or RL training loop. Derives from RLHF theoretically but is simpler to implement and often comparable in performance.

**The pattern in action:** A base Llama model will continue a prompt about making explosives because it is trained to predict plausible continuations of text. After SFT and RLHF, the same model refuses the same prompt. The pretraining parameters are largely unchanged; the behavior change came from alignment fine-tuning. This is why aligned models can be "jailbroken" by prompts that shift the distribution toward the base model's continuation behavior — the capability remains in the weights.

**Common traps:**
- Thinking alignment training fundamentally changes the model's knowledge or capabilities. It primarily changes the probability of certain types of outputs — the underlying capabilities remain largely intact.
- Thinking RLHF solves hallucination. RLHF optimizes for human preference, and human raters sometimes prefer confident, fluent hallucinations over honest expressions of uncertainty.

---

## 11. Inference Optimization

**What the interviewer is testing:** Whether you know the levers for making LLM inference faster and cheaper, and can reason about which bottleneck each addresses.

**The reasoning structure:** LLM inference has two phases with different bottlenecks. The prefill phase (computing attention over the full prompt) is compute-bound. The decoding phase (generating one token at a time) is memory-bandwidth-bound because it requires loading model weights for each step regardless of batch size — bandwidth limits how fast weights can move from memory to compute.

Optimization levers:

**Quantization:** reduce weight precision from float32/float16 to int8 or int4. Reduces memory footprint and bandwidth requirements. 8-bit quantization typically loses <1% on most benchmarks. 4-bit is more aggressive but acceptable with careful calibration (GPTQ, AWQ). The gain comes from fitting more of the model in faster memory.

**KV-cache:** cache key and value projections for prompt and previously generated tokens. Eliminates recomputation during decoding. Standard in all production deployments.

**Continuous batching:** rather than waiting for all requests in a batch to complete before processing new ones, process incoming requests continuously by inserting and removing sequences from the batch dynamically. Dramatically improves GPU utilization when request durations vary.

**Speculative decoding:** a small draft model generates $k$ tokens speculatively; the large model verifies all $k$ tokens in parallel in one forward pass. Accepted tokens are kept; the first rejected token triggers resampling from the large model. The speedup depends on acceptance rate — works well when the draft model closely matches the large model's distribution, achieving 2–3× speedup on common text.

**Model distillation:** train a smaller student model to mimic the large model's output distribution. The student deploys at much lower cost with acceptable quality degradation for many applications.

**The pattern in action:** A legal document processing system uses a 70B parameter model taking 8 seconds per document. The team applies 4-bit quantization (4× memory reduction, ~2× throughput), adds KV-caching (removes redundant computation on shared document preambles), and implements continuous batching (utilization from 40% to 85%). Total throughput improves by ~4×. Accuracy drops 0.8% on the internal benchmark — an acceptable tradeoff.

**Common traps:**
- Defaulting to "buy a bigger GPU" as the first optimization. Infrastructure costs scale linearly; quantization, batching, and speculative decoding improve algorithmic efficiency.
- Applying the same optimization regardless of bottleneck. Memory-bandwidth-bound decoding benefits from quantization and caching; compute-bound prefill benefits from batching and operator fusion. Diagnosing the bottleneck before applying an optimization is mandatory.

---

## 12. LLM Evaluation

**What the interviewer is testing:** Whether you understand that LLM evaluation is genuinely hard and that simple automated metrics often fail to capture what matters.

**The reasoning structure:** Standard classification metrics (accuracy, F1) apply when the output space is finite and well-defined. For open-ended text generation, the output space is enormous and equivalence is fuzzy. "The meeting is at 3 PM" and "The meeting starts at 15:00" are semantically equivalent but score zero overlap by exact string match.

**Standard benchmarks:** MMLU (world knowledge across 57 subjects), HellaSwag (commonsense reasoning), TruthfulQA (truthfulness), HumanEval (code generation accuracy). Each measures something specific. High benchmark performance does not guarantee good performance on a specific application.

**Perplexity:** measures how well the model assigns probability to a held-out text. Lower perplexity means the model assigns higher probability to the test distribution. It does not measure factual accuracy, helpfulness, or output quality.

**LLM-as-judge:** use a capable LLM to rate or compare model outputs on a rubric. Correlates well with human judgment on many tasks, scales well, but introduces judge bias — capable LLMs tend to prefer longer, more confident, or stylistically similar responses.

**Human evaluation:** the gold standard for subjective quality. Expensive, slow, and has variance from annotator disagreement. For production applications, the most valid signal is human feedback on real queries.

**Task-specific metrics:** precision/recall on information extraction, ROUGE for summarization, pass@k for code generation. Most reliable when the output space is structured enough to define correctness.

**The pattern in action:** A model scores 82% on MMLU but performs poorly on an internal customer service evaluation. The internal evaluation tests accuracy on domain-specific product knowledge not appearing in MMLU. Benchmark performance was a poor predictor of production quality because the distribution mismatch between the benchmark and the deployment domain is large.

**Common traps:**
- Treating perplexity as a measure of output quality. Low perplexity means the model fits the test distribution, not that it generates accurate or useful outputs.
- Treating benchmark leaderboard position as a proxy for production suitability without evaluating on data from the target domain.

---

## 13. Hallucination

**What the interviewer is testing:** Whether you understand what causes hallucination, why it is architecturally fundamental rather than a bug to be patched, and what mitigation strategies are available.

**The reasoning structure:** Hallucination occurs when a model generates plausible-sounding text that is not grounded in its training data or the provided context. It is not random noise — it is the model confidently producing text that satisfies the autoregressive objective (high probability given the prompt and previous tokens) but does not correspond to facts.

Sources of hallucination:
- **Knowledge gaps:** the model lacks the specific information and generates a plausible-sounding substitute
- **Conflation:** the model confuses similar entities or facts from training data
- **Context neglect:** the model ignores provided context and generates from parametric memory
- **Overconfidence:** the model assigns high probability to false statements — miscalibrated confidence

Why it is architecturally hard to eliminate: the training objective (predict the next token) does not penalize plausible false statements more than plausible true ones. The model is optimized to produce likely continuations, not to verify factual accuracy. RLHF partially addresses this by training on human feedback that penalizes false claims, but human raters also rate confident hallucinations favorably when they cannot detect the error.

Mitigation strategies:
- **RAG:** grounds responses in retrieved documents, reducing reliance on parametric memory for factual claims
- **Chain-of-thought prompting:** forces explicit reasoning steps, making errors more detectable and easier to check
- **Calibration fine-tuning:** train on examples where the model expresses uncertainty when it should, reducing confident wrong answers
- **Attribution requirements:** require the model to cite the source for each claim; claims without citations are flagged
- **Output verification:** pass generated claims through a separate fact-checking step before serving

**The pattern in action:** A model asked for a citation produces a plausible-looking academic reference (author names, title, journal, year) that does not exist. The model learned the format of citations from training data and generates a syntactically correct but semantically fabricated one — the next-token prediction objective rewards getting the format right, not the content. RAG with a citation database prevents this: the model can only cite documents it has retrieved, so citations are verifiable by construction.

**Common traps:**
- Framing hallucination as something that will be "fixed" in the next model version. The training paradigm is fundamentally susceptible — improvements reduce frequency but do not eliminate the underlying mechanism. System design should assume hallucination occurs and build in verification.
- Treating hallucination as a uniform problem. The rate and type vary dramatically by domain, query type, and context provision. A model that hallucinates at 5% rate on well-represented topics may hallucinate at 40% rate on obscure topics from the long tail of training data.

---

## Quick Diagnostics

**If asked when to use RAG vs fine-tuning:**
RAG is for knowledge freshness and grounding — when the model needs information it could not have seen or that changes over time. Fine-tuning is for behavior — when you need consistent changes to style, format, tone, domain fluency, or reasoning approach. Mixing these up is a common and expensive mistake.

**If asked how to reduce LLM latency:**
Identify the bottleneck first. Prefill-bound (long prompts)? Optimize prompt compression or reduce retrieved context size. Decode-bound (long generation)? Quantize, add speculative decoding, or use a smaller model. Memory-bandwidth-bound? Quantize weights or optimize KV-cache management.

**If asked about LLM reliability in production:**
Raise: hallucination rates, output format consistency, latency distribution, cost per query, context length limits, behavior on adversarial or out-of-distribution prompts, and monitoring strategy. A reliable LLM system is a system built around an LLM — with retrieval, verification, fallbacks, and monitoring — not just an LLM with a good system prompt.
