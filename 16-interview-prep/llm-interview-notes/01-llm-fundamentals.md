---
module: LLMs
topic: Interview Notes
subtopic: Llm Fundamentals
status: unread
tags: [llms, ml, interview-notes-llm-fundamenta]
---

> _Full study notes. For the quick-recall version, see [llm-fundamentals-snappy.md](llm-fundamentals-snappy.md)._

# LLM Fundamentals — Interview Notes

## What A Senior Answer Should Sound Like

When an interviewer asks about LLM fundamentals, do not stop at definitions. The stronger answer ties the concept to:

- the training objective
- the inference constraint
- the failure mode
- the production implication

That is the difference between knowing the vocabulary and being able to operate the system.

---

## Foundation Models

**The problem**: before 2018, every NLP task required a separate, fully supervised model trained from scratch. Sentiment analysis needed labeled sentiment data; translation needed parallel corpora; NER needed BIO-tagged sentences. You could not reuse a sentiment model for translation. The bottleneck was labeled data — a new task meant thousands of human-hours of annotation.

**The core insight**: if a model is trained to predict the next word on enough internet text, it must learn grammar, facts, reasoning patterns, and implicit structure as a side effect. That latent knowledge can be redirected to almost any task, often with zero additional labeled examples.

**The mechanics**: self-supervised pretraining on raw text (next-token prediction or masked token prediction), followed by task-specific adaptation via prompting or fine-tuning. The pretraining loss is free — no human annotation needed at scale. Adaptation is cheap because the base knowledge is already there.

**What breaks**: the model freezes its knowledge at training time (staleness), inherits biases from the pretraining corpus, and has no principled way to distinguish what it learned from what it hallucinated. Emergent capabilities appear unpredictably above certain parameter thresholds, making capability forecasting unreliable. A flaw in the foundation (bias, factual error, misaligned behavior) propagates to every downstream application.
At interview level, the key production implication is simple: better pretraining does not remove the need for retrieval, policy controls, and evals.

**What the interviewer is testing**: do you understand *why* transfer learning at this scale works — not just that it works?

**Common traps**: confusing "foundation model" with "LLM" (CLIP and Stable Diffusion are foundation models; they are not language models); claiming models "search a database" (they have no retrieval at inference unless explicitly built in); conflating foundation model with AGI.

---

## How LLMs Work: Autoregressive Next-Token Prediction

**The problem**: a neural network trained to classify sentiment cannot generate text. You need a different framing. The question is: what single training objective forces a model to learn language well enough to write, reason, and follow instructions?

**The core insight**: predict the next token given all preceding tokens. To predict well on diverse internet text, the model must learn syntax, semantics, world facts, discourse structure, and style simultaneously — because all of these affect what comes next. Generation becomes trivial: just keep predicting.

**The mechanics**: at each step, the model takes a sequence of token IDs, runs them through a stack of transformer layers, and outputs a probability distribution over the vocabulary via softmax. Sample (or greedily select) the next token, append it to the sequence, repeat. During training: compute cross-entropy loss between the predicted distribution and the true next token; backpropagate; update weights. The same model is used for training and inference — the causal mask ensures token t cannot attend to tokens t+1, t+2, ... so the training on all positions can be parallelized.

**What breaks**: hallucination — the model produces the most statistically plausible continuation, not the most factually accurate one. It cannot "decide not to answer" unless trained to do so. It cannot look ahead before generating — it is committed to each token as it produces it. Context window limits are hard: attention is O(N²) in sequence length.
The useful follow-up is: if the model is just predicting the next token, where do you inject truth, safety, and freshness? The answer is outside the base objective.

**What the interviewer is testing**: whether you understand that generation is just prediction on a loop, not "retrieval" or "thinking" in any classical sense.

**Common traps**: saying the model "looks up" answers; not knowing that base models will complete instructions as if generating more instructions (they need SFT to learn the instruction-following contract); confusing temperature with quality (higher temperature = more random, not more creative in any guaranteed sense).

---

## The Transformer Architecture

**The problem**: RNNs process sequences token-by-token, which means: (1) long-range dependencies degrade because information must travel through many sequential steps; (2) training cannot be parallelized because step t requires step t-1. In 2017, these were the two critical bottlenecks preventing scale.

**The core insight**: replace recurrence entirely with attention. If you want token i to relate to token j, compute that relationship directly as a dot product — no need for information to travel through intermediate steps. Because every pair can be computed in parallel as a matrix multiplication, you can use GPUs at full capacity.

**The mechanics**:

1. **Input embeddings**: map discrete token IDs to continuous vectors in R^d.
2. **Positional encoding**: attention is a set operation (permutation invariant). Add sinusoidal position signals to embeddings before processing, so the model can distinguish token order.
3. **Multi-head self-attention**: for each layer, project inputs into Q (query), K (key), V (value) matrices. Compute:
   ```
   Attention(Q, K, V) = softmax(QKᵀ / √d_k) · V
   ```
   Scale by √d_k because large d_k causes dot products to grow large, pushing softmax into regions with near-zero gradients. Run H heads in parallel; each learns different relationship types. Concatenate and project.
4. **Feed-forward network (FFN)**: applied independently at each position — two linear layers with a nonlinearity. This is where factual knowledge is predominantly stored.
5. **Residual connections + LayerNorm**: x → x + sublayer(x) lets gradients bypass layers during backprop, enabling hundreds of stacked layers without vanishing gradients. LayerNorm normalizes per token (not per batch), making it sequence-length independent.

**What breaks**: attention is O(N²) in sequence length — both memory and compute scale quadratically. A 1M-token context requires ~1 trillion attention weight pairs. This makes very long contexts expensive and drives research into sparse attention, linear attention, and state-space models. The FFN is also stateless — it stores no long-term memory between conversations.

**What the interviewer is testing**: can you derive why each component exists, rather than listing them?

**Common traps**: saying transformers generate all tokens at once (training is parallel; inference is still autoregressive one token at a time); confusing self-attention with cross-attention (cross-attention compares two different sequences, e.g., encoder output and decoder state); not knowing where facts are stored (FFN, not attention).

---

## Tokenization and BPE

**The problem**: neural networks operate on fixed-size numerical inputs. Text is variable-length and open-vocabulary — new words are coined daily, code uses arbitrary symbols, and non-English scripts have thousands of characters. Word-level tokenization explodes the vocabulary and fails on unseen words. Character-level tokenization makes sequences so long that context windows become useless.

**The core insight**: the right unit is somewhere between characters and words. Frequent sequences (common words) should be single tokens because they carry dense, reusable meaning. Rare sequences should be split into familiar subwords, preserving compositionality.

**The mechanics — BPE**:
1. Initialize vocabulary with all individual bytes (0–255). This guarantees zero out-of-vocabulary errors for any Unicode text.
2. Scan the training corpus and count all adjacent byte-pair frequencies.
3. Merge the most frequent pair into a new single symbol. Add to vocabulary.
4. Repeat from step 2, now treating the merged symbol as atomic. Stop at target vocabulary size (typically 32k–128k).

Result: common English words become single tokens; rare words decompose into known substrings; non-English text uses more tokens per word (the "token tax").

**What breaks**: tokenization is computed outside the neural network. The model never sees individual characters — only token IDs. This is why LLMs fail to count letters ("how many r's in strawberry?") — they receive "strawberry" as a single opaque token. Low-resource languages are penalized: if the tokenizer was trained on 99% English data, a Hindi word that would be 1 English-equivalent token becomes 6–10 byte tokens, consuming context and costing more.
In production, the token tax is also a budget tax: every extra token costs latency and money, so multilingual performance and cost are tied together.

**What the interviewer is testing**: do you know that tokenization is a lossy preprocessing step that has downstream consequences for model behavior, cost, and multilingual performance?

**Common traps**: assuming 1 token = 1 word (rule of thumb: 1 token ≈ ¾ of an English word); not knowing that the model cannot "see" subword spelling, explaining failures on spelling/character-counting tasks; confusing the tokenizer (a deterministic algorithm) with the model.

---

## Attention Mechanism in Depth

**The problem**: in a sentence like "The trophy didn't fit in the suitcase because it was too big," the word "it" is ambiguous. A model needs to figure out that "it" refers to "trophy" based on context. Any fixed-width convolution or simple lookup cannot resolve this — the relevant context is arbitrarily far away.

**The core insight**: let every token "query" every other token and weight its own representation by how much each other token is relevant. The relevance weights are learned, not fixed. This gives the model a dynamic, content-based routing mechanism that works regardless of distance.

**The mechanics**:

Each token i learns to ask: "which other tokens are relevant to me?" by computing a query vector Qᵢ. Each token j says "here's what I offer" via a key vector Kⱼ. Relevance of j to i is Qᵢ · Kⱼ. After softmax normalization, each token's output is a weighted sum of value vectors:

```
out_i = Σⱼ softmax(Qᵢ · Kⱼ / √d_k) · Vⱼ
```

Running H parallel attention heads lets the model simultaneously resolve different relationship types (e.g., syntactic agreement in one head, coreference in another). The output is concatenated and projected back to d_model.

KV cache: during autoregressive generation, K and V matrices for past tokens do not change. Caching them eliminates recomputation, making inference tractable. KV cache size scales as O(n_layers × n_heads × sequence_length × d_head) — this is the primary memory cost at inference for long contexts.

**What breaks**: attention cannot distinguish which relationships are causal versus correlational — it just learns what co-occurs in training data. Attending to all tokens equally is the right thing at train time (via teacher forcing) but expensive at inference. For 128k-token contexts, the KV cache alone can exceed GPU memory.
If context gets very long, the bottleneck is rarely "model intelligence" first. It is usually memory pressure and retrieval quality.

**What the interviewer is testing**: whether you understand why QKV is split into three projections rather than just doing raw dot-products, and what the KV cache is.

**Common traps**: not knowing that multi-head attention's outputs are concatenated (not averaged); confusing the scaling factor explanation (it's about softmax gradient saturation, not normalization per se); not knowing that the KV cache is what makes long-context inference expensive, not the model weights.

---

## Decoder-Only vs. Encoder-Only vs. Encoder-Decoder

**The problem**: the original transformer was designed for translation: encode the source sentence in one branch, decode the target in another. But different tasks have different information access requirements. A model doing classification needs to see the full input bidirectionally. A model generating text must not see the future.

**The core insight**: restrict what positions can attend to what, and you get fundamentally different model behaviors.

**The mechanics**:

- **Encoder-only (BERT)**: no causal mask — each token attends to all others in both directions. Excellent for classification, NER, embedding generation. Cannot generate text autoregressively (would need to "see" tokens it hasn't emitted yet).
- **Decoder-only (GPT family, Llama, Claude)**: causal mask — token i can only attend to tokens 1..i. Trained as a next-token predictor. Everything current state-of-the-art LLMs do uses this architecture because the pretraining objective (next-token prediction) and the architecture (causal attention) are perfectly aligned.
- **Encoder-decoder (T5, BART)**: encoder processes the full input bidirectionally; decoder generates output with cross-attention to the encoder's hidden states. Natural for tasks with explicit input/output structure (translation, summarization). More parameters to serve, slower inference.

**What breaks**: decoder-only models technically have access to their full generated history (the KV cache) but cannot revise what they have already emitted — no backtracking. Encoder-only models are not generative. Encoder-decoder models require running both branches at inference, increasing latency.
That is why architecture choice is usually a latency and training-objective decision, not just a modeling-preference decision.

**What the interviewer is testing**: whether you can reason about *why* modern LLMs are decoder-only, not just *that* they are.

**Common traps**: saying BERT "doesn't work for generation" without explaining why (the causal constraint, not architecture difficulty); confusing cross-attention (used in encoder-decoder) with self-attention; not knowing that BERT's bidirectionality is what makes it unsuitable for autoregressive text generation.

---

## Scaling Laws

**The problem**: training large models is expensive and irreversible. If you are spending $10M on a training run, you need to know in advance: how many parameters, how much data, and how long to train to get the best model for that budget?

**The core insight**: LLM performance (measured by validation loss) scales as a smooth power law in model size, data size, and compute. These relationships are empirically stable enough to extrapolate — you can run small experiments and predict large model behavior.

**The mechanics — Chinchilla scaling laws (Hoffmann et al., 2022)**:

For a fixed compute budget C (in FLOPs), the optimal model size N and token count D satisfy approximately:
```
N_optimal ≈ C^0.5
D_optimal ≈ C^0.5
```
with the approximate rule: D ≈ 20 × N (20 tokens per parameter). Earlier models (GPT-3) were significantly undertrained — massive models trained on too little data. Chinchilla showed that for the same compute, a smaller model trained on more data beats a larger model trained on fewer tokens.

**What breaks**: scaling laws predict loss on the pretraining distribution, not task performance or safety. "Emergent abilities" (tasks that are near-zero performance, then suddenly useful above a threshold) are not well captured by smooth power laws — they appear as phase transitions. Inference cost scales with parameter count even if training was optimal, creating a tension between training efficiency and serving cost.
So a staff-level answer always separates training economics from serving economics.

**What the interviewer is testing**: do you know that "bigger is always better" was disproven by Chinchilla, and why compute-optimal training matters?

**Common traps**: citing GPT-3 era intuition that bigger is always better; not knowing the 20-tokens-per-parameter rule of thumb; confusing training compute budget with inference cost.

---

## Sampling and Decoding Strategies

**The problem**: the model produces a probability distribution over its entire vocabulary at each step. How you select the next token from that distribution determines whether the output is coherent, creative, repetitive, or nonsensical.

**The core insight**: greedy decoding (always pick the most probable token) maximizes local probability but produces repetitive, degenerate text. True random sampling preserves diversity but produces incoherent text. The right approach truncates the distribution to only plausible tokens, then samples from that restricted set.

**The mechanics**:

- **Greedy**: argmax over vocabulary. Deterministic, fast, often produces repetition loops.
- **Temperature**: divide logits by T before softmax. T < 1 sharpens the distribution (more deterministic); T > 1 flattens it (more random). T = 0 approaches greedy.
- **Top-k**: sample only from the k most probable tokens. Problem: k is fixed regardless of how peaked or flat the distribution is.
- **Top-p (nucleus sampling)**: include the smallest set of tokens whose cumulative probability ≥ p (typically 0.9–0.95). The number of candidates adapts to the distribution's shape. Standard for most production deployments.
- **Beam search**: maintain B candidate sequences simultaneously, expanding the most probable ones. Better for constrained tasks (translation, summarization) but produces generic text in open-ended generation.

**What breaks**: top-p can still sample low-probability tokens if the distribution is flat (many equiprobable tokens). Repetition penalty hacks are often needed on top of sampling to prevent looping. Beam search produces text that scores high on likelihood metrics but reads as bland to humans.
For product use, the choice is usually not "best decoding algorithm" but "best decoding strategy for the user experience we want."

**What the interviewer is testing**: can you explain why greedy decoding is bad for quality, and what top-p actually does?

**Common traps**: confusing temperature = 0 with greedy (they are equivalent but not mechanically identical in all implementations); saying "higher temperature = better quality" (it increases diversity, which may or may not be better depending on the task); not knowing that beam search is not used for chat generation.

---

## Pretraining, SFT, and RLHF: The Three-Phase Pipeline

**The problem**: a model trained only on next-token prediction will complete "What is the capital of France?" with more quiz questions, not with "Paris." It has no concept of the user-assistant interaction contract. Even after instruction tuning, it may produce outputs that are technically responsive but harmful, sycophantic, or dishonest — because those patterns also exist in training data.

**The core insight**: each phase shapes a different dimension of behavior. Pretraining instills knowledge and capability. SFT teaches the interaction format. RLHF aligns the model to human preference in ways too subtle to specify exhaustively via examples.

**The mechanics**:

1. **Pretraining**: causal language modeling on hundreds of billions of tokens. The objective is to minimize cross-entropy loss on next-token prediction. Produces a "base model" with rich language capability but no instruction-following behavior.

2. **Supervised Fine-Tuning (SFT)**: train on (instruction, response) pairs using the same cross-entropy loss, but only backpropagate on the response tokens (not the instruction tokens). Teaches the user→assistant format. A few thousand high-quality pairs can produce a capable instruction-following model (LIMA result: 1,000 curated examples ≈ 50,000 noisy ones).

3. **RLHF**: collect human preference labels (response A vs. B for the same prompt). Train a reward model R(x, y) on these pairs. Then optimize the LLM to maximize expected reward while staying close to the SFT baseline:
   ```
   maximize E[R(prompt, response)] - β · KL(π || π_ref)
   ```
   The KL penalty prevents reward hacking — the policy gaming the reward model by exploiting patterns that score high but are not actually preferred by humans.

**What breaks**: RLHF requires a high-quality reward model, which is itself a trained model that can be wrong. Reward hacking occurs when the policy finds reward model exploits (e.g., verbose responses score higher if annotators equate length with quality). DPO (Direct Preference Optimization) eliminates the reward model entirely by reparameterizing the RLHF objective, producing equivalent alignment with a simple cross-entropy loss on preference pairs.
The practical interview answer is that alignment is still an engineering problem: data quality, preference design, and evals matter more than the acronym.

**What the interviewer is testing**: do you understand why all three phases are necessary and what each one actually changes?

**Common traps**: saying RLHF "teaches the model to be helpful" (SFT does that; RLHF refines the tradeoffs); not knowing that DPO achieves alignment without a separate reward model; confusing the SFT loss (applied only to response tokens) with pretraining loss (applied to all tokens).

---

## Hallucination

**The problem**: LLMs confidently produce statements that are factually wrong, invented, or unfalsifiable. This is not a bug caused by a specific failure — it is a structural property of how these models work. You need to understand why it happens to know when it is and is not fixable.

**The core insight**: the model is trained to produce the most plausible continuation of a sequence, not the most accurate one. Plausibility (what sounds right given the training distribution) and accuracy (what is true) are correlated but not identical. When the model has low or conflicting training signal about a fact, it fills the gap with a plausible-sounding invention.

**The mechanics**: hallucination occurs in at least three distinct patterns:

1. **Closed-book knowledge gaps**: the fact was absent or rare in training data. The model extrapolates from related patterns (e.g., correct author, wrong publication year).
2. **Sycophantic drift**: the model was RLHF-trained on human preference. Humans often prefer confident, fluent answers. The model learns to produce confident answers even when uncertain.
3. **Context infidelity**: in RAG or long-context settings, the model generates answers consistent with its parametric knowledge rather than the retrieved context. The context contradicts what was "learned," and the model discounts it.

**What breaks the mitigations**: RAG reduces knowledge-gap hallucination but not sycophancy or context infidelity. Chain-of-thought reduces reasoning errors but not factual invention. Grounding prompts ("answer only from context") reduce context infidelity but only when the model has been trained to follow such instructions reliably.
If the question is "can we trust the answer?", the answer is almost always "only after we define what kind of error we are defending against."

**What the interviewer is testing**: whether you understand that hallucination is structural, not a single fixable bug — and that different types require different interventions.

**Common traps**: saying "we can fix hallucination with better prompts" (for knowledge-gap cases, no); confusing hallucination with all model errors (out-of-date knowledge is a different problem — staleness); not knowing that confidence calibration is separate from accuracy.

---

## Context Window and Positional Encoding

**The problem**: attention requires every token to "know" its position in the sequence. Without positional information, "dog bites man" and "man bites dog" are the same set of tokens and produce the same attention patterns. But fixed sinusoidal encodings cannot extrapolate to sequence lengths longer than those seen in training.

**The core insight**: positions need to be encoded in a way that generalizes. Relative position (token A is 5 steps before token B) generalizes better than absolute position (token A is at index 5000) because relative offsets appear throughout training data regardless of document length.

**The mechanics**:

- **Sinusoidal (original Transformer)**: PE(pos, 2i) = sin(pos / 10000^(2i/d)), PE(pos, 2i+1) = cos(pos / 10000^(2i/d)). Different frequencies for different dimensions. Fixed, not learned. Cannot extrapolate beyond training sequence lengths.
- **Learned absolute positions**: each position gets a learned embedding. Works well within training range; degrades outside it.
- **RoPE (Rotary Position Embedding)**: applied to Q and K before the dot product. Relative position enters as a rotation in embedding space — token i and token j's attention score depends naturally on (i - j). Used in Llama, Mistral, and most modern open models. Can be extended via RoPE scaling.
- **ALiBi (Attention with Linear Biases)**: adds a linear bias to attention scores based on distance. Enables zero-shot extrapolation to longer sequences.

**What breaks**: even with long-context architectures, retrieval quality degrades for content placed in the middle of very long contexts ("lost in the middle"). Extending context windows does not solve the attention bottleneck — it just moves the threshold. Memory cost for KV cache grows linearly in sequence length per layer.

**What the interviewer is testing**: whether you understand why position encoding exists at all, not just which variants exist.

**Common traps**: not knowing that the original Transformer's sinusoidal encodings cannot extrapolate; confusing absolute and relative position encoding; saying "longer context = always better retrieval" when lost-in-the-middle is a real empirical effect.

---

## Mixture of Experts (MoE)

**The problem**: a dense transformer scales parameter count by making every layer larger. But at inference, every token activates every parameter — so doubling parameters doubles inference cost. Is there a way to have more total capacity (more parameters) without proportionally increasing inference cost?

**The core insight**: replace the dense FFN with multiple smaller FFNs (experts). For each token, a router selects only a small subset of experts to activate. Total parameter count is large; active parameter count per token stays constant.

**The mechanics**: each MoE layer has E expert FFNs and a learned router that outputs softmax probabilities over experts. The top-k experts (typically k=2) are selected per token, their outputs weighted by router probability and summed:

```
output = Σ_{i in top-k} router(x)_i · Expert_i(x)
```

A load-balancing auxiliary loss encourages uniform expert utilization — without it, a few experts dominate (expert collapse) and most parameters are wasted. GPT-4 is widely believed to use MoE; Mixtral-8x7B uses 8 experts with top-2 selection, giving 46.7B total parameters but 12.9B active per token.

**What breaks**: MoE models require loading all expert weights into memory even though only a subset fires per token — total memory footprint is the full parameter count, not the active parameter count. Expert collapse is a training stability problem. Load balancing loss adds a hyperparameter that interacts with the main training objective.

**What the interviewer is testing**: do you understand that "MoE has more parameters" and "MoE is cheaper to infer" can both be true simultaneously?

**Common traps**: confusing total parameters with active parameters; not knowing that MoE still requires full model memory (all experts must be on device or accessible quickly); saying that MoE improves quality for free — the quality gain requires careful load-balancing and hyperparameter tuning.


## Rapid Recall (Snappy Answers)

> _Quick-recall companion. For the full deep-dive, see [llm-fundamentals.md](01-llm-fundamentals.md)._

# LLM fundamentals — the “DevOps brain” edition

You already ship systems. So we’ll talk about LLMs like systems: **interfaces, failure modes, cost, latency, and guardrails**.

**How to use this file**
- **Direct answer**: what you say in 10 seconds.
- **Azure/DevOps bridge**: the mapping you already trust.
- **Analogy**: fashion / Mumbai Indians / classic romance + Gulzar-style context.
- **Mini prompt**: quick interactive break.

> **North star:** An LLM is a **probabilistic next-token service**. Production GenAI is mostly making that service behave.

---

# Q1: What are foundation models, and how have they changed AI engineering?
- **Direct answer:** A **foundation model** is a huge self-supervised model trained on broad data that you **adapt** (prompt/RAG/fine-tune) for many tasks.
- **Azure/DevOps bridge:** It’s “build on Azure” vs “build your own datacenter.” You spend less time on core modeling and more on **integration + policy + observability**.
- **Analogy:** Capsule wardrobe—same staples, different styling.
- **Mini pop quiz:** Name a non-text foundation model. → CLIP, Stable Diffusion, Wav2Vec.

---

# Q2: What is a Large Language Model (LLM), and how does it work?
- **Direct answer:** An LLM generates text by predicting the **next token** repeatedly (autoregressive decoding), typically using Transformers.
- **Mechanics (one breath):** prompt → tokenize → Transformer → logits → softmax → decode → append token → repeat.
- **Mini prompt:** `temperature=0` gives you what vibe? → deterministic / strict.

---

# Q3: What are Transformer Models and how do they work?
- **Direct answer:** Transformers use **self-attention** to connect tokens directly, enabling parallel training and long-range context.
- **Ghazal hook:** Like a Gulzar line—meaning depends on what came before and what surrounds it. Attention is “who is this word in conversation with?”
- **Trade-off:** naive attention has ugly scaling with long context.

---

# Q4: What are the key components of a Transformer model?
- **Direct answer:** embeddings + position signal + attention + FFN + residuals + LayerNorm.
- **DevOps bridge:** residual paths are the “safe rollback route” for gradients.

---

# Q5: What is tokenisation in LLMs?
- **Direct answer:** Tokenization slices text into tokens (often subwords) and maps them to IDs.
- **Practical truths:** **1 token ≠ 1 word**; tokenizers can create a **token tax** for some languages.

---

# Q6: Explain BPE (Byte Pair Encoding).
- **Direct answer:** BPE iteratively merges frequent adjacent symbols until a fixed vocabulary is built (often byte-level).
- **Fashion analogy:** threads → panels → outfit.

---

# Q7: Explain WordPiece and SentencePiece.
- **Direct answer:** WordPiece uses likelihood-based merges; SentencePiece is language-agnostic and treats spaces as characters.
- **Mini prompt:** Which is friendlier to languages without spaces? → SentencePiece.

---

# Q8: What is positional encoding, and why is it needed in Transformers?
- **Direct answer:** Attention doesn’t encode order by itself, so we inject position info (absolute or relative-ish like RoPE/ALiBi).

---

# Q9: What is causal masking?
- **Direct answer:** A training-time mask that prevents decoder models from attending to future tokens.
- **DevOps bridge:** blocks leakage at the mechanism level (like forbidding prod secrets in PR pipelines).

---

# Q10: What is self-attention, and how does it work in Transformers?
- **Direct answer:**

\[
Attention(Q,K,V)=softmax\left(rac{QK^T}{\sqrt{d_k}}
ight)V
\]

- **Why \(\sqrt{d_k}\)?** Keeps dot products from blowing up so gradients don’t vanish.

---

# Q11: Explain Query (Q), Key (K), and Value (V) in attention.
- **Direct answer:** Q asks, K advertises, V carries payload. Dot(Q,K) gives weights; weights mix V.

---

# Q12: What are multi-head attention mechanisms? Why use multiple attention heads?
- **Direct answer:** Multiple heads learn different relationship patterns in parallel.
- **MI analogy:** multiple fielders for different shots.

---

# Q13: What is the context window in LLMs, and why does it matter?
- **Direct answer:** Max tokens (input + output) per call; outside the window = forgotten.
- **DevOps bridge:** context window ≈ RAM; fine-tuning ≈ installed software; RAG ≈ read-through cache.

---

# Q14: What is temperature in the context of LLMs?
- **Direct answer:** Temperature rescales logits before softmax:

\[
p_i=rac{e^{x_i/T}}{\sum_j e^{x_j/T}}
\]

- Low T = deterministic; high T = diverse + more hallucination risk.

---

# Q15: Explain Top-p (nucleus) sampling and Top-k sampling.
- **Direct answer:** Top-k keeps the k most likely tokens; Top-p keeps the smallest set with cumulative probability ≥ p.

---

# Q16: What are logits, and how are they used in text generation?
- **Direct answer:** Logits are raw scores per vocab token; softmax turns them into probabilities; decoding chooses/samples a token.

---

# Q17: What are skip connections (residual connections) in Transformers?
- **Direct answer:** Add input back to sublayer output; improves stability and gradient flow.

---

# Q18: Open-source vs closed-source LLMs? When choose which?
- **Direct answer:** Closed = strongest, fastest adoption; Open = control/customization/on-prem, but you own ops.
- **Azure lens:** closed feels like managed PaaS; open feels like AKS.

---

# Q19: Encoder-only vs decoder-only vs encoder-decoder Transformer architectures?
- **Direct answer:** Encoder-only for understanding/embeddings; decoder-only for generation; encoder-decoder for seq2seq transforms.

---

# Q20: What is KV cache, and how does it speed up inference?
- **Direct answer:** Cache past K/V tensors so decoding doesn’t recompute history every token.
- **Infra reality:** it’s a latency win and a VRAM bill.

---

# Q21: Autoregressive vs masked language modeling.
- **Direct answer:** CLM predicts next token from past; MLM fills blanks using both sides.

---

# Q22: What is model distillation?
- **Direct answer:** Train a smaller student to mimic a larger teacher (often via soft targets/logits).

---

# Q23: What is Mixture of Experts (MoE)?
- **Direct answer:** Route tokens to a subset of expert networks; huge capacity with sparse compute.

---

# Q24: Dense vs sparse models?
- **Direct answer:** Dense runs all params each step; sparse conditionally activates subsets.

---

# Q25: What is Flash Attention?
- **Direct answer:** GPU-kernel optimization that reduces attention memory IO (tiling + fused ops) without changing math.

---

# Q26: What is Grouped-Query Attention (GQA)?
- **Direct answer:** Share K/V across groups of query heads to shrink KV cache with little quality loss.

---

# Q27: How does Rotary Position Embedding (RoPE) work?
- **Direct answer:** Rotate Q/K vectors by position so dot products encode relative distance; helps length extrapolation.

---

# Q28: LLM ignores instructions. How do you enforce structured output?
- **Direct answer:** Combine system constraints + few-shot + constrained decoding/structured outputs (schema-driven).

---

# Q29: Context window limit on long documents. What do you do?
- **Direct answer:** Use RAG, map-reduce summarization, and/or agentic search tools.

---

# Q30: Make it say “I don’t know.”
- **Direct answer:** Explicit escape hatch + grounding (retrieval) + low temperature + optional confidence gates.

---

# Q31: Code generation keeps adding explanations. What do you do?
- **Direct answer:** Tighten format contract (“only code”), few-shot, stop sequences/structured output, lower temperature.

---

# Q32: Explain the “Lost in the Middle” phenomenon.
- **Direct answer:** In long prompts, models use start/end info better than middle; recall can be U-shaped.

---

# Q33: What is quantization?
- **Direct answer:** Reduce weight/activation precision (FP16 → INT8/INT4) to cut memory and bandwidth.

---

# Q34: How do you choose between 8-bit, 4-bit, and 2-bit quantization?
- **Direct answer:** 8-bit ≈ near-lossless; 4-bit ≈ best local/cheap; 2-bit often collapses quality.

---

# Q35: AWQ vs GPTQ?
- **Direct answer:** GPTQ compensates quantization error layer-wise; AWQ protects salient weights via activation profiling.

---

# Q36: Explain PEFT.
- **Direct answer:** Fine-tune by training a tiny subset (adapters/LoRA/prefix) while freezing the base.

---

# Q37: How does LoRA work?
- **Direct answer:** Learn low-rank matrices \(A,B\) so \(\Delta Wpprox BA\), added to frozen weights.

---

# Q38: What is QLoRA?
- **Direct answer:** Quantize base model (e.g., 4-bit NF4) during tuning while training LoRA adapters in 16-bit.

---

# Q39: What is RLHF?
- **Direct answer:** Alignment using preference feedback (reward signal) + optimization (historically PPO).

---

# Q40: What is Constitutional AI?
- **Direct answer:** A rulebook-driven alignment approach using critique/rewrite against a “constitution,” reducing human labeling.

---

# Q41: What is instruction tuning?
- **Direct answer:** Supervised tuning on instruction→response pairs to make base models follow commands reliably.

---

# Q42: What is DPO?
- **Direct answer:** Preference alignment via a stable supervised-style objective on (chosen, rejected) pairs.

---

# Q43: What is PPO?
- **Direct answer:** RL algorithm with clipped updates to prevent destructive policy jumps.

---

# Q44: Why use reward models in RLHF?
- **Direct answer:** Humans can’t label at scale; reward models provide scalable scalar feedback.

---

# Q45: Name 3 metrics used to evaluate LLMs.
- **Direct answer:** Perplexity, task benchmarks (MMLU/HumanEval), judge/human A/B.

---

# Q46: What is perplexity?
- **Direct answer:** How surprised the model is on held-out text:

\[
PPL(X)=\exp\left(-rac{1}{N}\sum_{i=1}^{N}\log P(x_i\mid x_{<i})
ight)
\]

---

# Q47: Why is ROUGE or BLEU often insufficient for evaluating LLMs?
- **Direct answer:** n-gram overlap misses semantic equivalence; use semantic/functional/judge-based evals.
