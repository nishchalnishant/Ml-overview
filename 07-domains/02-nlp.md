---
module: Interview Prep
topic: Ml
subtopic: Nlp
status: unread
tags: [interviewprep, ml, ml-nlp]
---
# Natural Language Processing — Classical Foundations

**Primary reference:** [NLP methods](01-nlp-fundamentals.md) | For transformers, attention, tokenization, BERT/GPT, and PEFT, see [llm/02-nlp-transformers.md](03-nlp-transformers.md)

---

## 1. Why NLP Is Hard

The core difficulty is representational, not computational: meaning is context-dependent. Bag-of-words makes "man bites dog" identical to "dog bites man" — order is lost. Each major technique recovers something the previous one lost:

- Bag-of-words: loses word order.
- TF-IDF: still no order, but weights by discriminative value.
- Word embeddings: distributional similarity, but context-free ("bank" has one vector for finance and river).
- RNNs: process order, but lose long-range context to gradient decay.
- Transformers: attend across all positions at once, removing the sequential bottleneck (see [llm/02-nlp-transformers.md](03-nlp-transformers.md)).

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

**Trap:** say *why* Transformers won (sequential bottleneck + vanishing gradient, solved by direct position-to-position attention — see [llm/02-nlp-transformers.md §3](03-nlp-transformers.md)), not just "Transformers are better." RNNs still matter where O(n²) attention cost is prohibitive.

---

## 5. Perplexity

$$\text{PPL}(W) = \exp\left(-\frac1N\sum_i \log P(w_i\mid w_{<i})\right)$$

Perplexity $k$ means the model is as uncertain as choosing uniformly among $k$ options. Only comparable across models with the *same* tokenizer/vocabulary. Low perplexity doesn't imply good generation — a model can score well on training-adjacent text and still hallucinate on novel prompts.

**Trap:** using perplexity to compare models with different tokenizers, or treating perplexity gains as a proxy for downstream task quality.

---

## 6. Seq2Seq and Summarization

Encoder builds an input representation; decoder generates output autoregressively.

**Extractive summarization:** copies spans from source — faithful by construction, but choppy. **Abstractive:** generates new text — more natural, but can hallucinate.

Choose based on hallucination tolerance: legal/compliance leans extractive or grounded; news can tolerate abstraction.

**Trap:** ROUGE (n-gram overlap) rewards copying and can't detect hallucination or reward good paraphrase — treat high ROUGE with skepticism.

---

## 7. Stemming vs Lemmatization

Both normalize word forms for grouping. Stemming: mechanical suffix-chopping, fast but crude ("universal" → "univers"). Lemmatization: dictionary-aware, respects part of speech, slower but more accurate.

Modern transformer pipelines mostly don't need either (subword tokenization handles morphology), but they still matter in classical pipelines and keyword search.

---

## 8. Dependency Parsing

Identifies typed grammatical relationships (subject, object, modifier) as a tree over a sentence — useful when you need to know "who did what to whom" precisely (e.g. structured info extraction, semantic role labeling). Transformers implicitly encode some syntax in attention patterns, but explicit parsers still matter when you need guaranteed, verifiable structured output.

---

## Quick Diagnostics

**Why Transformers won:** attention handles long-range dependencies with O(n²) per-layer cost instead of forcing information through O(n) sequential hidden states — removes the training bottleneck and lets any position access any other directly. Full mechanism: [llm/02-nlp-transformers.md](03-nlp-transformers.md).

**Choosing an approach for a new NLP task:** ask data size, latency budget, generative vs discriminative, interpretability need — in that order — before picking an architecture. TF-IDF + logistic regression is a legitimate production answer under tight data/latency/interpretability constraints.
