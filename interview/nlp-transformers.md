# NLP and Transformers

This is the fast NLP interview file.

The main story is simple:

language modeling moved from counting words to modeling context.

And once attention arrived, the whole field changed shape.

---

## 1. Tokenization

You turn raw text into units the model can process.

Common approaches:

- word-level
- subword
- byte-level

Subword methods like BPE matter because they balance:

- vocabulary size
- rare word handling
- sequence length

---

## 2. TF-IDF vs Embeddings

TF-IDF:

- sparse
- count-based
- no true semantics

Embeddings:

- dense
- learned
- capture similarity better

That distinction comes up constantly.

---

## 3. Self-Attention

Self-attention lets one token weigh other tokens in the same sequence.

That is why context becomes dynamic rather than fixed.

It is the core mechanism behind Transformers.

---

## 4. Why Scale by `sqrt(d_k)`?

Without scaling, dot products grow too large as dimension grows.

That makes softmax too sharp and gradients less healthy.

So the scaling keeps training behavior more stable.

---

## 5. BERT vs GPT

BERT:

- encoder-only
- better for understanding tasks

GPT:

- decoder-only
- better for generation

If you say that cleanly, you are already ahead of many answers.

---

## 6. MLM vs Causal LM

MLM:

- masked language modeling
- predict hidden tokens

Causal LM:

- next-token prediction
- left-to-right generation

That contrast is interview gold.
