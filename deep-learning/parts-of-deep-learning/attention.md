# Attention

Attention is the mechanism that let deep learning stop reading sequences like a forgetful intern and start understanding context more flexibly.

It is one of the biggest reasons modern NLP and LLMs look the way they do.

---

# 1. What Attention Does

Attention lets one token decide which other tokens matter more.

That is the whole idea.

Instead of compressing everything into one hidden state, the model dynamically pulls the most relevant context when needed.

---

# 2. Query, Key, Value

Best way to explain them:

- Query = what this token is looking for
- Key = what other tokens offer for matching
- Value = the information they provide if matched

That is enough for most interviews.

---

# 3. Scaled Dot-Product Attention

Core formula:

- `softmax(QK^T / sqrt(d_k)) V`

Meaning:

- compare queries and keys
- normalize scores
- use them to mix values

Why scale by `sqrt(d_k)`?

To stop large dot products from making softmax too sharp and training too unstable.

---

# 4. Self-Attention vs Cross-Attention

## Self-Attention

Tokens attend to other tokens in the same sequence.

Used for:

- context building
- internal sequence understanding

## Cross-Attention

Tokens in one sequence attend to another sequence.

Used for:

- encoder-decoder setups
- multimodal interaction
- retrieval-conditioned generation

---

# 5. Multi-Head Attention

Instead of one attention pattern, use multiple heads in parallel.

Why?

Because language and data contain multiple kinds of relationships at once.

Different heads can learn:

- local structure
- long-range links
- syntax
- coreference

**Cricket analogy**

It is like watching the same over with different camera angles:

- one shows field placement
- one shows batter movement
- one shows bowler release

Same moment.
Different signal.
