---
module: Interview Prep
topic: Ml
subtopic: Nlp
status: unread
tags: [interviewprep, ml, ml-nlp]
---
# Natural Language Processing

**Primary reference:** [NLP methods](../../03-deep-learning/methods/nlp-fundamentals.md) | [LLM fundamentals](../../05-llms/interview-notes/llm-fundamentals.md)

---

## 1. Why NLP Is Hard

**What the interviewer is testing:** Whether you understand that the core difficulty is representational, not computational. Language resists simple encoding because meaning is context-dependent in ways that raw features cannot capture.

**The reasoning structure:** Start with the failure of the obvious approach. Represent text as a bag of word counts and "man bites dog" becomes the same vector as "dog bites man." That failure reveals the real problem: you need to encode not just which words appear, but their relationships and order. Every major NLP technique is a different answer to that same underlying problem.

The history of NLP can be read as a sequence of attempts to recover what each previous representation lost:
- Bag-of-words loses word order entirely
- TF-IDF loses order but recovers relative discriminative value across documents
- Word embeddings capture distributional similarity but produce context-free representations — "bank" gets one vector regardless of whether the context is financial or riverbank
- RNNs process order sequentially but lose long-range context to gradient decay
- Transformers model relationships between all positions simultaneously, removing the sequential bottleneck

Each step recovered something the previous approach lost. Understanding this progression is more valuable in an interview than memorizing only the endpoint.

**The pattern in action:** A candidate who says "Transformers replaced RNNs because they are better" is describing the outcome. A candidate who says "RNNs required information to travel through sequential hidden states, which bottlenecks both training parallelism and long-range signal; attention removes that bottleneck by connecting any two positions directly" is describing the mechanism — and that is what the interviewer actually wants.

**Common traps:** Treating NLP as solved because LLMs exist. Large language models handle many tasks well but fail predictably on rare domains, low-resource languages, structured reasoning, and factual precision. Knowing where the hard part still lives distinguishes strong candidates from those who pattern-match on buzzwords.

---

## 2. Bag-of-Words and TF-IDF

**What the interviewer is testing:** Whether you can reason about the tradeoffs of simple representations — when they are good enough and when they break — and whether you know to reach for the simplest working solution first.

**The reasoning structure:** Bag-of-words is not wrong; it is incomplete. For many classification tasks where word identity matters more than order — distinguishing "this ticket is a complaint about billing" from "this ticket is a feature request" — bag-of-words works surprisingly well. The first question should always be: does order matter for this task? If not, the simpler model may be the right choice.

TF-IDF improves on raw counts by recognizing that a word appearing in every document tells you nothing about which document is which. It down-weights ubiquitous terms and up-weights terms that are rare globally but frequent in a specific document — those are the discriminative ones.

$$\text{tf-idf}(t, d) = \text{tf}(t, d) \cdot \text{idf}(t)$$

where:

$$\text{tf}(t, d) = \frac{\text{count of } t \text{ in } d}{\text{total tokens in } d}, \quad \text{idf}(t) = \log \frac{N}{\text{df}(t) + 1}$$

The $+1$ prevents division by zero for unseen terms. "The" gets near-zero IDF and thus near-zero weight regardless of frequency. A domain-specific term appearing in 3 of 10,000 documents gets high IDF and high weight when it appears in a target document.

**The pattern in action:** You are building a support-ticket classifier for a cloud platform. TF-IDF plus logistic regression often achieves 85–90% accuracy on well-defined categories, trains in seconds, and is fully interpretable — you can read the weights to see which terms drive each class prediction. Before reaching for a fine-tuned transformer, this baseline is worth running. If it hits the accuracy target, you ship it. If not, you now have a calibrated sense of how much the more expensive model needs to improve things.

**Common traps:**
- Using TF-IDF on tasks requiring understanding of syntax or negation. It cannot distinguish "the server is not responding" from "the server is responding."
- Skipping the simple baseline because it feels unsophisticated. Production systems that are interpretable, cheap, and accurate enough are better engineering decisions than complex systems that are marginally more accurate.

---

## 3. Word Embeddings

**What the interviewer is testing:** Whether you understand what embeddings are learning — distributional similarity — and why that is useful but not sufficient for all tasks.

**The reasoning structure:** The distributional hypothesis states that words appearing in similar contexts tend to have similar meanings. Word embeddings operationalize this by learning dense vectors where proximity in vector space corresponds to contextual similarity. This was a major advance because it gave downstream models useful prior structure: the model does not need to learn from scratch that "car" and "automobile" are related.

**Word2Vec Skip-gram objective:** predict surrounding context words from a center word.

$$J = \frac{1}{T} \sum_{t=1}^{T} \sum_{-c \leq j \leq c, j \neq 0} \log P(w_{t+j} | w_t)$$

where $P(w_o | w_c) = \frac{\exp(\mathbf{u}_o^T \mathbf{v}_c)}{\sum_{w=1}^{W} \exp(\mathbf{u}_w^T \mathbf{v}_c)}$

Computing the full softmax over the vocabulary is expensive. Negative sampling approximates it by sampling $K$ noise words and treating the problem as binary classification:

$$J_{\text{NEG}} = \log \sigma(\mathbf{u}_o^T \mathbf{v}_c) + \sum_{k=1}^{K} \mathbb{E}_{w_k \sim P_n(w)}[\log \sigma(-\mathbf{u}_{w_k}^T \mathbf{v}_c)]$$

**GloVe** factorizes the global log co-occurrence matrix:

$$J = \sum_{i,j=1}^{V} f(X_{ij}) \left(\mathbf{w}_i^T \tilde{\mathbf{w}}_j + b_i + \tilde{b}_j - \log X_{ij}\right)^2$$

where $f(x) = \min(1, (x/x_{\max})^{3/4})$ down-weights very frequent co-occurrences so they do not dominate.

**FastText** builds vectors from character n-grams:

$$\mathbf{v}_w = \sum_{g \in \mathcal{G}_w} \mathbf{z}_g$$

For "where" with $n=3$: the representation is the sum of vectors for {whe, her, ere} plus the full word token. A model trained without "unboxing" can still embed it from {unb, nbo, box, oxi, xin, ing}.

**The pattern in action:** You are deploying a product search system for a retail catalog that adds thousands of new product names each week. FastText is better here than Word2Vec because new product names like "AquaPure X7 Filter" can be embedded from subword components rather than mapping to unknown-token. This directly prevents the cold-start representation problem without retraining.

**Common traps:**
- Treating Word2Vec embeddings as context-sensitive. They are not. "Bank" always gets the same vector regardless of surrounding context. That is precisely why contextual embeddings were necessary.
- Assuming GloVe outperforms Word2Vec because it uses global statistics. On small corpora, the global co-occurrence matrix is too sparse to be informative, and Word2Vec's local window approach can perform better.

---

## 4. RNNs, LSTMs, and Why They Gave Way to Transformers

**What the interviewer is testing:** Whether you can explain architectural decisions as solutions to specific, nameable problems — not just name architectures and say one is better.

**The reasoning structure:** The core failure of vanilla RNNs is the vanishing gradient. When you backpropagate through 50+ timesteps, the gradient at each step is multiplied by the recurrent weight matrix. If the spectral radius of that matrix is below 1, the gradient signal shrinks exponentially toward zero. If above 1, it explodes. Either way, the model fails to learn long-range dependencies.

LSTMs address this with a gating mechanism that allows the cell state to propagate relatively unchanged over many timesteps:
- **Input gate:** how much new information to write to cell state
- **Forget gate:** how much of existing cell state to preserve — can stay near 1.0 for many steps
- **Output gate:** how much of cell state to expose as hidden state

Because the forget gate can remain near 1.0, gradients can flow backward through time without vanishing. GRUs achieve similar results with fewer parameters by merging some gates.

But even with LSTMs, training is inherently sequential — you cannot compute the hidden state at step $t$ until step $t-1$ is complete. This prevents full parallelization across the sequence dimension and limits how well information from early positions survives to late positions.

**The pattern in action:** For sentiment analysis on 20-word product reviews, an LSTM with attention is competitive with more expensive models. For a task requiring understanding across 2,000-word documents, the sequential bottleneck becomes a hard limit — both in wall-clock training speed and in representation quality, because the hidden state at position 2,000 has a degraded version of signal from position 5.

**Common traps:**
- Saying "RNNs are bad, Transformers are good" without articulating the specific failure. The interviewer wants the mechanism: sequential bottleneck plus vanishing gradient, solved by direct position-to-position routing in attention.
- Forgetting that RNNs and LSTMs remain relevant in resource-constrained deployment where the O(n²) attention cost of Transformers is prohibitive.

---

## 5. Attention and the Transformer

**What the interviewer is testing:** Whether you can derive the intuition behind attention — not just recite the formula — and explain why it changed what was architecturally possible.

**The reasoning structure:** The core idea is to replace the sequential bottleneck of a fixed hidden state with direct, learned routing between all positions. Instead of compressing the entire prefix into one vector, attention lets each position query every other position and weight them by relevance. The information path from any token to any other is now a constant number of steps regardless of sequence length.

The Query-Key-Value framing gives the mechanism a clean interpretation:
- A position emits a **query**: what am I looking for?
- Every position emits a **key**: what do I offer for matching?
- The dot product of query and key gives a compatibility score
- The **value** is what actually gets aggregated when the match is strong

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

**Why divide by $\sqrt{d_k}$?** Without scaling, when $d_k$ is large the dot products grow large in magnitude, pushing the softmax into saturation where gradients vanish. If $q_i, k_i \sim \mathcal{N}(0,1)$, then $\text{Var}(q^T k) = d_k$. Dividing by $\sqrt{d_k}$ normalizes variance to 1 regardless of embedding dimension.

Multi-head attention runs $h$ independent attention operations on projected subspaces and concatenates:

$$\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O$$
$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

Different heads learn to attend to different relationship types simultaneously — syntactic structure, semantic similarity, positional proximity, coreference. These specializations emerge from training, not from explicit supervision.

**The pattern in action:** In named entity recognition, the word "Washington" must resolve its ambiguity — person, city, or state — from context potentially several tokens away. An LSTM's hidden state at that position has a diluted version of the earlier context. An attention head can directly route from "Washington" to "traveled to" and "gave a speech" with high weight, disambiguating cleanly in a single operation.

**Common traps:**
- Explaining $\sqrt{d_k}$ as preventing "too large values" without specifying why that matters. The answer is softmax saturation and gradient vanishing, not numerical overflow.
- Describing attention as "the model looks at all words." Attention is learned and task-specific. The model does not uniformly attend to all positions — it learns which positions are relevant for each query.

---

## 6. BERT vs GPT — Encoder vs Decoder

**What the interviewer is testing:** Whether you understand that the architecture choice follows from the pretraining objective, which follows from the intended downstream use — not from arbitrary design choices.

**The reasoning structure:** Both are Transformer-based but differ in the attention mask applied during pretraining.

**BERT (encoder-only):** bidirectional attention — every token can attend to every other token. Pretraining task is masked language modeling: predict randomly masked tokens using both left and right context. This is ideal for tasks where you have the full input at inference time (classification, NER, extractive QA) because bidirectional context makes each position's representation richer.

**GPT (decoder-only):** causal (left-to-right) attention — each token can only attend to previous tokens. Pretraining task is next-token prediction. This is the only sensible objective when the model must generate tokens it has not yet produced. You cannot use future tokens to predict the current one during generation, so the model must not see them during training either.

Neither is universally better — the architecture choice is driven by the task. Using BERT for open-ended generation fights the architecture. Using GPT for classification works but does not leverage the bidirectional representation capacity BERT was designed to provide.

**The pattern in action:** You are building a system to classify customer intent from a support message. Use a fine-tuned BERT-style encoder — you have the full message at inference time and bidirectional context produces richer representations. You are building a conversational assistant that generates multi-turn responses. Use a GPT-style decoder — you need autoregressive generation, and causal masking is a requirement of the task, not a limitation.

**Common traps:**
- Asserting "BERT is better for understanding" as a hard rule. GPT-class models at sufficient scale are highly effective for understanding tasks; the preference is a tendency, not a law.
- Ignoring encoder-decoder architectures (T5, BART), which are often the right choice for sequence-to-sequence tasks where you need both a rich input encoding and autoregressive output generation.

---

## 7. Tokenization and BPE

**What the interviewer is testing:** Whether you understand tokenization as a design choice with real downstream consequences, not just a preprocessing detail.

**The reasoning structure:** The core tension is vocabulary size versus sequence length. Large vocabularies produce shorter sequences and better coverage of common words, but require large embedding tables and handle rare words poorly. Character-level tokenization handles every possible word but produces very long sequences that strain O(n²) attention. BPE navigates this by finding a data-driven middle ground.

**BPE algorithm:**
1. Initialize with individual characters plus a special end-of-word token
2. Count all adjacent symbol pairs in the corpus
3. Merge the most frequent pair into a new symbol
4. Repeat until vocabulary size $V$ is reached (typically 32k–100k)

Result: common words become single tokens, rare words decompose into recognizable subword pieces, and novel words still receive reasonable coverage from their components.

**WordPiece** (BERT): merges pairs that maximize language model likelihood rather than raw frequency. Continuation subwords are prefixed with `##`.

**SentencePiece** (T5, LLaMA): operates on raw bytes, requires no language-specific preprocessing, handles any script.

| Vocab size | Sequence length | Novel word handling | Embedding table |
| :--- | :--- | :--- | :--- |
| Small (8k) | Longer | Better (more subword splitting) | Smaller |
| Large (100k+) | Shorter | Fewer splits of known words | Larger |

**The pattern in action:** A model fine-tuned on English prose with a 32k-token vocabulary deployed on code containing identifiers like `getUserAuthTokenFromCacheOrRefresh` will over-split these into many fragments, inflating sequence length and fragmenting semantic units. A tokenizer trained on or adapted to code handles this substantially better — which is why code models use domain-specific tokenizers.

**Common traps:**
- Confusing tokenization with the embedding lookup. Tokenization converts text to integer token IDs. The embedding layer then maps IDs to vectors. These are separate operations.
- Assuming the tokenizer is fixed. Vocabulary extension, tokenizer fine-tuning, and full retraining are all valid approaches for domain adaptation.

---

## 8. Perplexity

**What the interviewer is testing:** Whether you can explain what perplexity measures, what its limits are, and when it is and is not a useful evaluation signal.

**The reasoning structure:** Perplexity is the exponentiated average negative log-likelihood:

$$\text{PPL}(W) = \exp\left(-\frac{1}{N} \sum_{i=1}^{N} \log P(w_i | w_1, \ldots, w_{i-1})\right)$$

A perplexity of $k$ means the model is as uncertain at each prediction step as if it were choosing uniformly among $k$ options. Lower is better, and the number has direct meaning: perplexity 50 is meaningfully better at predicting the test distribution than perplexity 200.

But perplexity has hard limits as an evaluation signal. It is only comparable across models using the same tokenizer and vocabulary — comparing perplexity between a 32k-token model and a 100k-token model is technically invalid. More critically, low perplexity does not imply good generation quality. A model can assign high probability to training-adjacent text while hallucinating freely on novel prompts.

**The pattern in action:** You are comparing two pretrained checkpoints to select a starting point for fine-tuning. Perplexity on a held-out sample from your target domain is a useful signal — the checkpoint with lower domain perplexity has learned representations closer to your distribution and will likely require less fine-tuning. But if you are choosing between checkpoints for customer-facing summarization quality, you need human evaluation or a task-specific metric, not perplexity.

**Common traps:**
- Comparing perplexity across models with different tokenizers. This produces misleading conclusions.
- Treating perplexity reduction as a proxy for downstream task improvement. They often correlate but do not have to — a model can improve perplexity through memorization without improving reasoning or generalization.

---

## 9. Seq2Seq and Summarization

**What the interviewer is testing:** Whether you understand the extractive-abstractive tradeoff and how to choose based on application requirements rather than which approach sounds more advanced.

**The reasoning structure:** Sequence-to-sequence models take one sequence and produce another of different length. The encoder builds a representation of the input; the decoder generates the output autoregressively from that representation plus previously generated tokens.

For summarization, two fundamentally different approaches exist:
- **Extractive:** copy spans directly from the source document. Faithful by construction because no new words are introduced. Tends to produce choppy summaries because natural summaries often require paraphrase.
- **Abstractive:** generate new text. More natural-sounding. Introduces the risk of hallucination — the model can produce plausible-sounding claims not present in the source.

The right choice depends on the application's tolerance for hallucination. Legal summarization should probably be extractive or include explicit grounding verification. News summarization can tolerate more abstraction.

**The pattern in action:** A customer service team wants automated summaries of call transcripts attached to support tickets. Extractive summarization is the safer choice — the summary consists of direct quotes from the transcript, which supervisors can verify. Abstractive might produce cleaner prose but introduces the risk that a summary slightly misrepresents what was said, which matters in a dispute resolution context.

**Common traps:**
- Treating high ROUGE scores as evidence of good summarization. ROUGE measures n-gram overlap with reference summaries, which rewards systems that copy text. It systematically undervalues abstractive quality and cannot detect hallucination.
- Not raising the evaluation problem: reference summaries reflect one annotator's judgment about what is important, which may not match what the application actually needs.

---

## 10. Stemming vs Lemmatization

**What the interviewer is testing:** Whether you know where classical preprocessing still matters and can explain the practical difference between two approaches that look similar but trade off differently.

**The reasoning structure:** Both convert word surface forms to a base form for the purpose of grouping related words. Stemming applies mechanical heuristic rules — chop suffixes until the result looks like a root. Fast and simple, but crude: "universal" might stem to "univers." Lemmatization uses vocabulary knowledge to map to the actual dictionary form, respecting part of speech.

In modern transformer-based pipelines, neither is typically necessary because subword tokenization handles morphological variation at the representation level. In classical NLP pipelines, search engines, and keyword-matching systems, the distinction remains meaningful.

**Common traps:** Treating them as interchangeable. Stemming trades linguistic accuracy for speed; lemmatization trades speed for correctness. Knowing which tradeoff to make and when is the substance of the answer.

---

## 11. Dependency Parsing

**What the interviewer is testing:** Whether you understand where structural analysis adds value and when transformers have subsumed it.

**The reasoning structure:** Dependency parsing identifies typed grammatical relationships between words — subject, object, modifier — forming a tree over the sentence. This matters when the application needs to understand who did what to whom with precision. A query "who acquired LinkedIn?" benefits from knowing "Microsoft" is the syntactic subject of "acquired" and "LinkedIn" is the direct object.

Modern fine-tuned transformers often implicitly encode syntactic structure in their attention patterns. But for tasks requiring guaranteed structured output — information extraction pipelines, semantic role labeling, generating structured queries from natural language — explicit parsers remain relevant because they produce interpretable, verifiable structure.

**Common traps:** Either calling dependency parsing obsolete (it has important applications in structured prediction and formal information extraction) or saying it is necessary for all NLP (transformers handle most tasks without it).

---

## Quick Diagnostics

**If asked why Transformers won:**
Attention handles long-range relationships in O(n²) operations per layer rather than requiring information to travel through O(n) sequential hidden states. This removes the sequential training bottleneck and allows each position to directly access any other — the capability that recurrent architectures fundamentally could not provide without unacceptable information loss.

**If asked to choose an approach for a new NLP task:**
Ask first: What is the data size? What is the latency budget? Is the task generative or discriminative? What is the interpretability requirement? TF-IDF plus logistic regression is a legitimate production answer when data is limited, latency is strict, and interpretability matters. Fine-tuned transformers are the answer when those constraints do not bind.
