**Primary reference:** [NLP methods](../../03-deep-learning/methods/nlp.md) | [LLM fundamentals](../../05-llms/interview-notes/llm-fundamentals.md)

# Natural Language Processing

NLP is really about one big question:

How do you turn messy human language into something a machine can reason over?

And as with poetry, lyrics, and ghazals, the answer is:

context changes everything.

---

# 1. Why NLP Is Hard

Language is slippery.

The same word can mean different things depending on:

- what came before
- what comes after
- tone
- domain
- intent

That is why NLP evolved from:

- counting words

to:

- modeling relationships and context

---

# 2. Bag-of-Words

Bag-of-Words represents text by counting words and ignoring order.

It is simple.
Fast.
Often strong as a baseline.

But it loses:

- syntax
- sequence
- nuance

So:

- "man bites dog"
- "dog bites man"

can look identical.

That is clearly not ideal.

---

# 3. TF-IDF

TF-IDF improves on Bag-of-Words by down-weighting very common words and up-weighting words that are more distinctive in a document.

$$\text{tf-idf}(t, d) = \text{tf}(t, d) \cdot \text{idf}(t)$$

where:

$$\text{tf}(t, d) = \frac{\text{count of } t \text{ in } d}{\text{total tokens in } d}, \quad \text{idf}(t) = \log \frac{N}{\text{df}(t) + 1}$$

- $N$: total number of documents
- $\text{df}(t)$: number of documents containing term $t$
- The $+1$ in the denominator prevents division by zero for unseen terms

**Intuition:** "the" appears in all documents → IDF ≈ 0 → weight ≈ 0. A domain-specific term in 3 of 10,000 documents → high IDF → high weight if also frequent in the target document.

**Variants:**
- Sublinear TF: $1 + \log(\text{tf})$ to dampen very high counts
- Smooth IDF: $\log\left(\frac{N+1}{\text{df}(t)+1}\right) + 1$ (sklearn default)
- BM25 (retrieval): adds document length normalization

```python
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features=10000, sublinear_tf=True)
X = tfidf.fit_transform(corpus)   # sparse matrix (n_docs × vocab)
```

TF-IDF captures term importance better than plain counts, but it still does not understand meaning or context.

---

# 4. Word Embeddings

Word embeddings represent words as dense vectors instead of sparse one-hot encodings.

Why that matters:

- similar words end up near each other
- semantic relationships become learnable
- models can generalize better

This was a huge step up from simple count-based methods.

---

# 5. Word2Vec vs GloVe vs FastText

## Word2Vec

Learns word vectors from local context prediction. Two architectures:

**Skip-gram:** predict context words from center word. Objective (maximize):

$$J = \frac{1}{T} \sum_{t=1}^{T} \sum_{-c \leq j \leq c, j \neq 0} \log P(w_{t+j} | w_t)$$

where $P(w_o | w_c) = \frac{\exp(\mathbf{u}_o^T \mathbf{v}_c)}{\sum_{w=1}^{W} \exp(\mathbf{u}_w^T \mathbf{v}_c)}$

Skip-gram works better for infrequent words; CBOW is faster.

**CBOW:** predict center word from context words. Averages context embeddings: $\hat{\mathbf{v}} = \frac{1}{2c}\sum_{-c \leq j \leq c, j\neq 0} \mathbf{v}_{w_{t+j}}$, then softmax.

**Negative sampling** (practical approximation):

$$J_{\text{NEG}} = \log \sigma(\mathbf{u}_o^T \mathbf{v}_c) + \sum_{k=1}^{K} \mathbb{E}_{w_k \sim P_n(w)}[\log \sigma(-\mathbf{u}_{w_k}^T \mathbf{v}_c)]$$

Avoids computing the full softmax denominator over the entire vocabulary. $K=5$–$20$ negative samples.

## GloVe

Factorizes the log co-occurrence matrix. Objective:

$$J = \sum_{i,j=1}^{V} f(X_{ij}) \left(\mathbf{w}_i^T \tilde{\mathbf{w}}_j + b_i + \tilde{b}_j - \log X_{ij}\right)^2$$

where $X_{ij}$ = co-occurrence count, $f(x) = \min(1, (x/x_{\max})^{3/4})$ down-weights very frequent pairs.

GloVe captures global corpus statistics; Word2Vec uses local context windows.

## FastText

Represents each word as a bag of character n-grams. For word $w$ with n-gram set $\mathcal{G}_w$:

$$\mathbf{v}_w = \sum_{g \in \mathcal{G}_w} \mathbf{z}_g$$

For example, "where" with $n=3$: {whe, her, ere} plus the full word token.

Handles OOV words: any unseen word can still get a vector from its n-grams. Especially useful for morphology-rich languages (German, Finnish, Turkish).

**Short interview line**

FastText is stronger when subword information matters because it builds representations from character n-grams rather than whole-word IDs alone.

---

# 6. Sequence Models: RNNs, LSTMs, GRUs

Before Transformers took over, sequence models were the stars of NLP.

## RNN

Processes tokens one by one.

Problem:

- struggles with long-range dependencies
- hard to parallelize

## LSTM / GRU

Improved versions with gating.

They decide:

- what to remember
- what to forget
- what to pass forward

**Ghazal analogy**

In a Gulzar verse, one word can quietly echo three lines later.
A plain RNN may forget the setup.
An LSTM is better at carrying the emotional memory forward.

That is the entire point.

---

# 7. Why Transformers Replaced RNNs

Transformers removed the sequential bottleneck.

Instead of processing one token at a time, they use attention so tokens can interact across the sequence more directly.

This gives:

- better parallelization
- better long-range context handling
- better scaling with data and compute

**Short answer**

Transformers replaced RNNs because attention handles long-range relationships better and allows efficient large-scale parallel training.

---

# 8. Attention in Plain English

Attention lets the model decide which other words matter most for understanding the current word.

That is what makes contextual meaning possible.

**Poetic analogy**

In an old romantic Bollywood song, a single word like "raat" or "dil" does not carry the same feeling in every line.
Its weight depends on the words around it.

Attention is the mathematical version of that sensitivity.

---

# 9. Query, Key, Value

- Query = what the current token is looking for
- Key = what other tokens offer for matching
- Value = the information they pass along if matched

Mathematically:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

- $QK^T \in \mathbb{R}^{n \times n}$: pairwise similarity matrix (each query against all keys)
- $/ \sqrt{d_k}$: prevents dot products from growing large with dimension, keeping softmax non-degenerate
- Softmax row: attention weights summing to 1 for each query position
- $\times V$: weighted combination of value vectors

**Multi-head attention** runs $h$ attention heads in parallel on projected subspaces:

$$\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O$$

$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

Each head can attend to different positions and representation subspaces simultaneously.

---

# 10. BERT vs GPT

## BERT

Encoder-style.
Built for understanding.

Usually better for:

- classification
- NER
- retrieval-style understanding

## GPT

Decoder-style.
Built for generation.

Usually better for:

- next-token generation
- chat
- completion
- creative or structured output

**Short answer**

BERT is optimized more for understanding full context; GPT is optimized for autoregressive generation.

---

# 11. Pretraining and Fine-Tuning

Pretraining gives the model broad language ability.

Fine-tuning adapts that ability to:

- a domain
- a task
- a product style

This is one of the biggest reasons modern NLP became so effective.

You no longer need to learn everything from scratch for every task.

---

# 12. Tokenization and BPE

You cannot feed raw text directly into the model.
You tokenize it first.

Why not tokenize by full words only? Vocabulary becomes huge (millions), rare words break things.

Why not tokenize by single characters? Sequences become too long; meaning is harder to learn.

**BPE (Byte Pair Encoding)** algorithm:

1. Initialize vocabulary with individual characters + a special end-of-word token
2. Count all adjacent symbol pairs in the corpus
3. Merge the most frequent pair into a new symbol
4. Repeat until vocabulary size $V$ is reached (typically 32k–100k)

Example: starting from characters {b, u, g, s, e, r}, after enough merges: "bugs" → ["bug", "s"], "bugger" → ["bug", "g", "er"].

**WordPiece** (BERT): similar to BPE but maximizes language model likelihood of the training data rather than frequency. Marks continuation subwords with `##`.

**SentencePiece** (T5, LLaMA): language-agnostic, treats the raw byte stream — no pre-tokenization step needed, handles any language.

**Unigram LM tokenizer** (XLNet, mBART): maintains a probability over all subword segmentations; picks the most probable one at inference.

**Vocabulary size tradeoffs:**

| Vocab size | Sequence length | OOV handling | Memory |
| :--- | :--- | :--- | :--- |
| Small (8k) | Longer | Better | Lower |
| Large (100k+) | Shorter | Rare tokens split less | Higher embedding table |

**Short answer**

BPE balances vocabulary size and sequence length by breaking rare words into reusable subword pieces.

---

# 13. Perplexity

Perplexity measures how surprised a language model is by actual text:

$$\text{PPL}(W) = P(w_1, w_2, \ldots, w_N)^{-1/N} = \exp\left(-\frac{1}{N} \sum_{i=1}^{N} \log P(w_i | w_1, \ldots, w_{i-1})\right)$$

where $N$ is the number of tokens. Equivalently: $\text{PPL} = \exp(H)$ where $H$ is the cross-entropy on the test set.

**Intuition:** a perplexity of $k$ means the model is as confused as if it had to pick uniformly among $k$ options at each step.

- PPL = 1: perfect prediction
- PPL = vocabulary size: random guessing

Lower perplexity means the model predicts the sequence better.

**Limitations:**
- Comparable only across models with the same tokenizer and vocabulary
- Low perplexity ≠ quality for open-ended generation — a model can be low-PPL and still hallucinate
- Not meaningful for tasks like instruction following or summarization — use task-specific metrics (BLEU, ROUGE, human preference)

**Bits per character (BPC):** $\text{BPC} = \log_2(\text{PPL}_{char})$ — tokenizer-agnostic alternative useful for cross-model comparison.

---

# 14. Stemming vs Lemmatization

## Stemming

Crude chopping of word endings.

Fast, but rough.

## Lemmatization

Maps words to cleaner dictionary base forms.

More linguistically informed.

This matters more in classical NLP pipelines than in large transformer pipelines, but it is still good interview knowledge.

---

# 15. Dependency Parsing

Dependency parsing identifies grammatical relationships between words.

Examples:

- subject
- object
- modifier

It matters when syntax is central to the problem.

Modern Transformers reduce how often you need explicit parsing pipelines, but the concept still matters.

---

# 16. Summarization

There are two main styles:

## Extractive

Select key sentences or spans from the original text.

Pros:

- safer
- easier to control

## Abstractive

Generate new summary text.

Pros:

- more flexible
- more natural

Cons:

- can hallucinate

That contrast is worth remembering.

---

# 17. Seq2Seq Models

Seq2Seq means sequence-to-sequence.

Input sequence goes in.
Output sequence comes out.

Examples:

- translation
- summarization
- question answering

Historically this meant encoder-decoder RNNs.
Now it often means encoder-decoder Transformers.

Same family idea.
Different engine.

---

# 18. t-SNE for NLP

t-SNE is mostly for visualization.

In NLP, it is often used to plot:

- word embeddings
- document embeddings
- cluster structure

It is useful for qualitative inspection.

Not for serious downstream modeling.

Very important distinction.

---

# Quick Thought Experiment

You are building a support-ticket classifier for Azure incidents.

Would you start with:

- a giant custom Transformer trained from scratch
- TF-IDF plus a simple classifier
- or a pretrained language model fine-tuned lightly

The smart answer is:

start with the strongest practical baseline for the data size and latency budget, not the most fashionable architecture on social media.

---

# Mini Pop Quiz

Why did Transformers win?

Best short answer:

Because they model context better at scale and remove the sequential training bottleneck of RNNs.

Short.
Correct.
Elegant.
