# Transformers

Transformers are what happened when deep learning decided:

We are done pretending sequential bottlenecks are charming.

They became dominant because they scale beautifully and handle context far better than older sequence models.

---

# 1. Why Transformers Won

RNNs process tokens one by one.

That creates two problems:

- slower training
- weaker long-range memory

Transformers solve this with attention.

That gives:

- better parallelization
- stronger context handling
- easier scaling with data and compute

---

# 2. Three Core Ideas

## Self-Attention

Tokens look across the sequence to build contextual meaning.

## Multi-Head Attention

Multiple attention patterns run in parallel.

## Positional Information

The model still needs order information, because attention alone does not inherently encode sequence order.

Those three ideas are the backbone.

---

# 3. Encoder-Only vs Decoder-Only vs Encoder-Decoder

## Encoder-Only

Best for understanding tasks.

Examples:

- BERT-like models

## Decoder-Only

Best for generation.

Examples:

- GPT-style models

## Encoder-Decoder

Best for transformation tasks.

Examples:

- translation
- summarization
- seq2seq generation

That triad comes up constantly.

---

# 4. Transformer Tradeoffs

Why they are great:

- scalable
- strong context modeling
- dominant in modern NLP and LLMs

Why they are painful:

- attention cost grows badly with sequence length
- large models are expensive
- they often need huge data or strong pretraining

So the right answer is not "Transformers are best."

It is:

> "Transformers are powerful, but their compute and data demands are a major design consideration."
