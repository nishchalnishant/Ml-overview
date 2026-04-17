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

This makes it very useful for:

- search
- document ranking
- classical text classification

**Short answer**

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

Learns word vectors from local context prediction.

Fast and influential.

## GloVe

Uses global co-occurrence statistics.

Great for capturing broader corpus structure.

## FastText

Breaks words into subword pieces.

Especially useful for:

- rare words
- morphology-rich languages
- out-of-vocabulary handling

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

Best explained simply:

- Query = what the current token is looking for
- Key = what other tokens offer for matching
- Value = the information they pass along if matched

That is enough for most interviews.

Do not turn it into a ceremony.

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

Why not tokenize by full words only?

Because:

- vocabulary becomes huge
- rare words break things

Why not tokenize by single characters only?

Because:

- sequences become too long
- meaning becomes harder to learn efficiently

So subword methods like **BPE** are the practical middle ground.

**Short answer**

BPE balances vocabulary size and sequence length by breaking rare words into reusable subword pieces.

---

# 13. Perplexity

Perplexity measures how surprised a language model is by actual text.

Lower perplexity usually means the model predicts the sequence better.

Useful for:

- comparing language models under the same setup

Less useful as the only real-world metric if your task is:

- instruction following
- summarization
- human preference alignment

Because low perplexity is not the same as being genuinely helpful.

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
