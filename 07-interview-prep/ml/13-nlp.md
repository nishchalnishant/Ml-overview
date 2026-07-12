---
module: Interview Prep
topic: Ml
subtopic: Nlp
status: unread
tags: [interviewprep, ml, ml-nlp]
---
# Natural Language Processing

**Primary reference:** [NLP methods](../../03-deep-learning/methods/01-nlp-fundamentals.md) | [LLM fundamentals](../../05-llms/interview-notes/01-llm-fundamentals.md) | [self-attention derivation](../llm/09-nlp-transformers.md)

---

## 1. Why NLP Is Hard

The core difficulty is representational, not computational: meaning is context-dependent. Bag-of-words makes "man bites dog" identical to "dog bites man" — order is lost. Each major technique recovers something the previous one lost:

- Bag-of-words: loses word order.
- TF-IDF: still no order, but weights by discriminative value.
- Word embeddings: distributional similarity, but context-free ("bank" has one vector for finance and river).
- RNNs: process order, but lose long-range context to gradient decay.
- Transformers: attend across all positions at once, removing the sequential bottleneck.

**Trap:** assuming LLMs make NLP "solved" — they still fail predictably on rare domains, low-resource languages, structured reasoning, and factual precision.

---

## 2. Bag-of-Words and TF-IDF

Bag-of-words works well when word identity matters more than order (e.g. ticket category classification). TF-IDF improves on raw counts by down-weighting terms common across all documents and up-weighting terms that are rare globally but frequent in a specific doc:

$$\text{tf-idf}(t,d) = \text{tf}(t,d)\cdot\text{idf}(t), \quad \text{idf}(t) = \log\frac{N}{\text{df}(t)+1}$$

TF-IDF + logistic regression is a legitimate fast, cheap, interpretable baseline (often 85-90% accuracy on well-defined categories) worth trying before a fine-tuned transformer.

**Trap:** TF-IDF can't handle negation/syntax ("server is not responding" vs "server is responding"). Don't skip the simple baseline just because it feels unsophisticated.

---

## 3. Word Embeddings

Distributional hypothesis: words in similar contexts have similar meaning. Embeddings learn dense vectors where proximity = contextual similarity.

**Word2Vec (skip-gram):** predict context words from a center word; uses negative sampling to avoid a full softmax over the vocabulary.

**GloVe:** factorizes the global co-occurrence matrix directly.

**FastText:** builds word vectors from character n-grams, so out-of-vocabulary words ("unboxing") still get a reasonable embedding from subword pieces — useful for catalogs with constantly-changing vocabulary.

**Traps:** Word2Vec/GloVe embeddings are context-free — "bank" always gets the same vector (this is why contextual embeddings/transformers were needed). GloVe isn't automatically better than Word2Vec — on small corpora the co-occurrence matrix is too sparse.

---

## 4. RNNs, LSTMs, and Why Transformers Replaced Them

Vanilla RNNs suffer vanishing/exploding gradients over long sequences — backprop through many timesteps repeatedly multiplies by the recurrent weight matrix.

LSTMs fix this with gates: **input** (what to write), **forget** (what to keep — can stay near 1.0 for long stretches, letting gradients flow), **output** (what to expose). GRUs do similar with fewer parameters.

Even LSTMs are sequential — hidden state at $t$ needs state at $t-1$ — which blocks parallelization and limits how well early-sequence signal survives to later positions.

**Trap:** say *why* Transformers won (sequential bottleneck + vanishing gradient, solved by direct position-to-position attention), not just "Transformers are better." RNNs still matter where O(n²) attention cost is prohibitive.

---

## 5. Attention and the Transformer

Attention replaces the fixed-hidden-state bottleneck with direct, learned routing between all positions — constant path length between any two tokens regardless of sequence length.

Query/Key/Value: each position emits a query ("what am I looking for") and a key ("what do I offer"); dot product gives a compatibility score; the value is what gets aggregated.

$$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Divide by $\sqrt{d_k}$ because dot products grow with dimension ($\text{Var}(q^Tk)=d_k$), which would push softmax into saturation and kill gradients.

Multi-head attention runs $h$ attention operations on projected subspaces in parallel and concatenates — different heads specialize in different relationship types (syntax, semantics, position) without explicit supervision.

**Trap:** explain $\sqrt{d_k}$ via softmax saturation/gradient vanishing, not "prevents large numbers." Attention weights are learned and task-specific, not "looking at everything uniformly."

---

## 6. BERT vs GPT — Encoder vs Decoder

Both are Transformers; they differ in the attention mask during pretraining.

**BERT (encoder-only):** bidirectional attention, masked-language-model pretraining. Good when the full input is available at inference (classification, NER, extractive QA).

**GPT (decoder-only):** causal (left-to-right) attention, next-token-prediction pretraining. Required for generation, since future tokens can't be seen during training or inference.

Encoder-decoder models (T5, BART) are often right for seq2seq tasks needing both a rich input encoding and autoregressive output.

**Trap:** "BERT is always better for understanding" is a tendency, not a law — large GPT-class models are strong at understanding too.

---

## 7. Tokenization and BPE

Tradeoff: vocabulary size vs sequence length. Large vocab → shorter sequences, worse rare-word coverage, bigger embedding table. Character-level → full coverage, very long sequences.

**BPE:** start from characters, iteratively merge the most frequent adjacent pair, until vocab size $V$ (typically 32k-100k) is reached.

**WordPiece** (BERT): merges by language-model likelihood rather than raw frequency. **SentencePiece** (T5, LLaMA): operates on raw bytes, no language-specific preprocessing needed.

**Trap:** a tokenizer trained on English prose over-splits code identifiers — domain-specific tokenizers matter. Tokenization (text → token IDs) is separate from the embedding lookup (IDs → vectors).

---

## 8. Perplexity

$$\text{PPL}(W) = \exp\left(-\frac1N\sum_i \log P(w_i\mid w_{<i})\right)$$

Perplexity $k$ means the model is as uncertain as choosing uniformly among $k$ options. Only comparable across models with the *same* tokenizer/vocabulary. Low perplexity doesn't imply good generation — a model can score well on training-adjacent text and still hallucinate on novel prompts.

**Trap:** using perplexity to compare models with different tokenizers, or treating perplexity gains as a proxy for downstream task quality.

---

## 9. Seq2Seq and Summarization

Encoder builds an input representation; decoder generates output autoregressively.

**Extractive summarization:** copies spans from source — faithful by construction, but choppy. **Abstractive:** generates new text — more natural, but can hallucinate.

Choose based on hallucination tolerance: legal/compliance leans extractive or grounded; news can tolerate abstraction.

**Trap:** ROUGE (n-gram overlap) rewards copying and can't detect hallucination or reward good paraphrase — treat high ROUGE with skepticism.

---

## 10. Stemming vs Lemmatization

Both normalize word forms for grouping. Stemming: mechanical suffix-chopping, fast but crude ("universal" → "univers"). Lemmatization: dictionary-aware, respects part of speech, slower but more accurate.

Modern transformer pipelines mostly don't need either (subword tokenization handles morphology), but they still matter in classical pipelines and keyword search.

---

## 11. Dependency Parsing

Identifies typed grammatical relationships (subject, object, modifier) as a tree over a sentence — useful when you need to know "who did what to whom" precisely (e.g. structured info extraction, semantic role labeling). Transformers implicitly encode some syntax in attention patterns, but explicit parsers still matter when you need guaranteed, verifiable structured output.

---

## Quick Diagnostics

**Why Transformers won:** attention handles long-range dependencies with O(n²) per-layer cost instead of forcing information through O(n) sequential hidden states — removes the training bottleneck and lets any position access any other directly.

**Choosing an approach for a new NLP task:** ask data size, latency budget, generative vs discriminative, interpretability need — in that order — before picking an architecture. TF-IDF + logistic regression is a legitimate production answer under tight data/latency/interpretability constraints.
