# Large Language Model

These answers focus on the version you can say in an interview: what the component does, why it matters, and what tradeoff it introduces.

---

# Q1: What is a Large Language Model (LLM), and how does it work?

**Interview-ready answer**

An LLM is a large neural network, usually transformer-based, trained on massive text corpora to predict tokens from context. At a high level, it converts text into tokens, maps those tokens into embeddings, processes them through many transformer blocks, and outputs a probability distribution over the next token. By repeating that next-token prediction process, it can generate text, code, and structured outputs. The key interview point is that an LLM is fundamentally a conditional sequence model, not a database lookup system.

---

# Q2: What are Transformer Models and how do they work?

**Interview-ready answer**

Transformers are neural architectures built around attention rather than recurrence. They represent tokens as embeddings, inject position information, and then repeatedly apply self-attention and feed-forward layers with residual connections and normalization. Their big advantage is that they model long-range dependencies effectively and train efficiently at scale because they process many tokens in parallel.

---

# Q3: What are the key components of a Transformer model?

**Interview-ready answer**

The core components are token embeddings, positional information, multi-head self-attention, feed-forward networks, residual connections, normalization layers, and an output projection into vocabulary logits. In an encoder-decoder transformer you also have cross-attention between encoded inputs and decoder states. The strong interview answer is to explain how these pieces work together: attention mixes context, feed-forward layers transform representations, and residual plus normalization make deep training stable.

---

# Q4: What is self-attention, and how does it work in Transformers?

**Interview-ready answer**

Self-attention lets each token compute a weighted summary of other tokens in the same sequence. It does that by projecting tokens into queries, keys, and values, comparing query-key similarity scores, normalizing those scores, and then using them to weight the value vectors. This is what allows the representation of a token to depend on relevant context elsewhere in the sequence.

---

# Q5: How does attention help capture long-range dependencies?

**Interview-ready answer**

Attention gives every token a direct path to every other token, so the model does not have to pass information step by step through a long recurrent chain. That makes it much easier to connect distant words, resolve references, and model relationships across long contexts. In contrast, recurrent models often struggle because the signal has to travel through many sequential updates.

---

# Q6: What does each Transformer block learn?

**Interview-ready answer**

Each transformer block refines the token representations. The attention sublayer learns what contextual information each token should pull in from elsewhere, while the feed-forward sublayer transforms that mixed information non-linearly. Across many layers, early blocks often capture local syntax or lexical patterns, while deeper blocks encode higher-level semantics, task structure, and reasoning-like patterns.

---

# Q7: What is pre-training vs fine-tuning in LLMs?

**Interview-ready answer**

Pre-training is the large-scale self-supervised phase where the model learns broad language patterns from huge unlabeled corpora, usually via next-token prediction. Fine-tuning is the adaptation phase where the pretrained model is adjusted for a narrower objective, such as instruction following, domain language, or a downstream task. Pre-training builds general capability; fine-tuning makes that capability useful for a specific behavior or domain.

---

# Q8: What are some challenges in training LLMs?

**Interview-ready answer**

The big challenges are compute cost, data quality, optimization stability, long training times, distributed systems complexity, and evaluation. There are also alignment and safety concerns, because a model can be statistically powerful but still produce harmful, biased, or misleading outputs. A strong answer should mention that training is not just a scaling problem; it is also a data curation, systems, and evaluation problem.

---

# Q9: What is zero-shot learning in the context of LLMs?

**Interview-ready answer**

Zero-shot learning means the model performs a task from the prompt alone without task-specific fine-tuning examples. The reason this works is that pretraining and instruction tuning give the model general patterns it can reuse across many tasks. The deeper point is that zero-shot performance depends heavily on prompt clarity, model capability, and how close the requested task is to patterns the model already learned.

---

# Q10: How do you handle bias and fairness in LLMs?

**Interview-ready answer**

I would treat bias mitigation as a lifecycle issue rather than only a post-processing step. That includes data curation, safety tuning, evaluation across groups and prompt styles, refusal policies, human review for high-risk use cases, and monitoring after deployment. It is also important to define the harm you care about, because bias can show up as stereotyping, unequal error rates, toxic generation, or differential performance across dialects and languages.

---

# Q11: What are some real-world applications of LLMs in business and tech?

**Interview-ready answer**

Common applications include customer support, search assistance, enterprise knowledge retrieval, code assistance, document extraction, summarization, workflow automation, and analytics over unstructured text. The best interview answer adds that most production systems are not "LLM alone"; they combine prompting with retrieval, tools, rules, and monitoring to make outputs reliable and useful.

---

# Q12: How does the Transformer architecture improve LLM performance over RNNs?

**Interview-ready answer**

Transformers improve over RNNs mainly through attention and parallelism. Attention gives direct access to long-range context, while parallel token processing makes large-scale training feasible. RNNs carry information sequentially through hidden state, which makes long dependencies harder to learn and training slower to scale. That is why transformers became the dominant architecture for modern LLMs.

---

# Q13: Explain the attention mechanism in LLMs.

**Interview-ready answer**

Attention is the mechanism that lets the model decide which parts of the context matter for producing the next representation or next token. Each token computes attention scores to other tokens, normalizes them, and uses those weights to aggregate information. The result is a context-aware representation in which meanings can shift based on surrounding text.

---

# Q14: What are multi-head attention mechanisms? Why use multiple attention heads?

**Interview-ready answer**

Multi-head attention runs several attention operations in parallel on different learned projections of the same input. The intuition is that different heads can specialize in different kinds of relationships, such as local syntax, coreference, or long-range dependency. Using multiple heads gives the model richer contextual mixing than a single attention map.

---

# Q15: Explain Query (Q), Key (K), and Value (V) in attention.

**Interview-ready answer**

Queries represent what a token is looking for, keys represent what each token offers for matching, and values represent the information that can be retrieved if there is a match. Attention scores are computed from query-key similarity, and the resulting weights are used to mix the values. This is why Q, K, and V are useful: they separate matching from information transport.

---

# Q16: Tokenization in Large Language Models (LLMs).

**Interview-ready answer**

Tokenization converts raw text into units the model can process, such as words, subwords, or bytes. It matters because it defines the model's effective vocabulary, context length usage, and handling of rare words, multilingual text, and code. In practice, modern LLMs often rely on subword or byte-level tokenization to balance vocabulary size with flexibility.

---

# Q17: What is subword tokenization?

**Interview-ready answer**

Subword tokenization breaks text into pieces that are smaller than words but larger than characters. This helps because full-word vocabularies become too large and fail on rare or unseen words, while character-level tokenization is often too long and inefficient. Subword methods let models represent frequent words efficiently while still handling rare words compositionally.

---

# Q18: What is BPE (Byte Pair Encoding) in LLMs?

**Interview-ready answer**

BPE is a tokenization method that starts from small units and repeatedly merges the most frequent adjacent pairs to build a vocabulary of common subword pieces. The result is a compact token set that captures frequent patterns while still allowing unseen words to be broken into known parts. The important interview point is that BPE is a tradeoff between vocabulary efficiency and sequence length.

---

# Q19: What is positional embedding in LLMs?

**Interview-ready answer**

Attention itself is permutation-invariant, so the model needs some way to know token order. Positional embeddings or positional encodings inject that order information into the token representations. Without them, the model could know which words are present but not their sequence. Modern systems may use learned position embeddings, rotary embeddings, or other relative-position schemes.

---

# Q20: What is temperature in the context of LLMs?

**Interview-ready answer**

Temperature is a decoding parameter that rescales logits before sampling. Lower temperature makes the output distribution sharper and more deterministic, while higher temperature makes it flatter and more diverse. It does not change the underlying model knowledge; it changes how aggressively the model samples from the probability distribution at inference time.

---

# Q21: What is causal masking?

**Interview-ready answer**

Causal masking prevents a token from attending to future tokens during training or generation. This is essential for autoregressive language modeling because the model should only use past context when predicting the next token. Without causal masking, the training objective would leak the answer from the future and the model would not learn proper next-token prediction behavior.

---

# Q22: What are skip connections?

**Interview-ready answer**

Skip connections, or residual connections, add the input of a layer back to its output. This stabilizes training by making it easier for gradients and information to flow through deep networks. In transformers, residual paths are a major reason the architecture can scale to many layers without becoming impossible to optimize.

---

# Q23: What is normalization?

**Interview-ready answer**

Normalization rescales activations so training is more stable. In LLMs, LayerNorm and related variants are used to control activation magnitudes, improve optimization, and support deeper models. The strongest interview answer is not "it normalizes the data" but "it stabilizes training dynamics inside the network."

---

# Q24: What is dropout, and how is it applied in LLMs?

**Interview-ready answer**

Dropout is a regularization technique where parts of the network are randomly zeroed during training so the model does not rely too heavily on specific paths. In transformers it can be applied to embeddings, attention weights, or feed-forward outputs depending on the implementation. Its role is to improve generalization, although very large modern models sometimes use less dropout than older architectures because scale and data already provide substantial regularization.

---

# Q25: Why does Attention use Softmax?

**Interview-ready answer**

Softmax converts raw attention scores into normalized positive weights that sum to one, which makes them easy to interpret as relative importance across tokens. It also sharpens differences between higher and lower scores, helping the model focus selectively. The important nuance is that softmax is a convenient differentiable weighting mechanism, not the only possible attention normalization.

---

# Q26: What does a vector database (Vector DB) store for LLM usage?

**Interview-ready answer**

A vector database stores dense embeddings, usually along with metadata and references back to the original source documents or chunks. In LLM systems, it is used for semantic retrieval: given a query embedding, the system finds nearby vectors that correspond to relevant content. The key point is that the vector DB does not store "the model's memory" in the same way model weights do; it stores external retrievable representations.

---

# Q27: How do you improve inference speed in production LLM deployments?

**Interview-ready answer**

I would improve inference speed through a combination of model-level and system-level changes: smaller or distilled models, quantization, optimized kernels, batching, prompt caching, retrieval to reduce prompt size, speculative decoding, and better serving architecture. The right answer depends on what the bottleneck is, such as GPU memory, token generation speed, or network overhead. In practice, prompt design and system architecture often matter as much as raw model size.

---

# Q28: What is the context window in LLM?

**Interview-ready answer**

The context window is the maximum amount of tokenized input and generated text the model can consider at once. It matters because anything outside that window is not directly visible to the model during inference. A larger context window helps with long documents and multi-step tasks, but it increases compute cost and does not automatically guarantee perfect long-context reasoning or retrieval.

---

# Q29: Explain Prompting, Retrieval-Augmented Generation (RAG), and Fine-Tuning.

**Interview-ready answer**

Prompting changes behavior at inference time by giving the model better instructions, examples, or structure. RAG supplements the prompt with retrieved external knowledge so the model can ground its answer in up-to-date or domain-specific information without changing the weights. Fine-tuning changes the model parameters so it behaves differently by default. A strong interview answer explains when to use each: prompting for fast iteration, RAG when knowledge grounding matters, and fine-tuning when you need consistent behavior, style, or domain adaptation that prompting alone cannot provide.
