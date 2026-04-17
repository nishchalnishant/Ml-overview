# Deep Learning for NLP

NLP changed dramatically once deep learning stopped treating text like a bag of disconnected words and started treating it like structured context.

That single shift changed everything.

---

# 1. The Evolution

Broadly, NLP evolved like this:

- Bag-of-Words and TF-IDF
- static embeddings
- sequence models like RNN/LSTM/GRU
- Transformers

That journey is worth understanding because interviewers often ask not only what won, but why it won.

---

# 2. Embeddings

Embeddings gave words dense vector representations instead of sparse IDs.

That allowed models to learn similarity and meaning much better.

This was the bridge from symbolic-feeling NLP to richer representation learning.

---

# 3. RNNs, LSTMs, GRUs

These models process sequences step by step.

Their strength:

- natural sequence handling

Their weakness:

- slow training
- trouble with long-range context

**Ghazal analogy**

If the emotion of the current line depends on three lines before it, the model needs memory.
That is exactly why LSTMs mattered before Transformers took over.

---

# 4. Transformers in NLP

Transformers won because attention handles context more flexibly and training scales far better.

That made them ideal for:

- large corpora
- transfer learning
- foundation models

In other words:

they removed the sequential bottleneck and opened the door to modern LLMs.
