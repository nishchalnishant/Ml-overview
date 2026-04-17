# Natural Language Processing (NLP)

This hub tracks the evolution of NLP from classical statistical methods to modern Transformer-based architectures. A senior candidate should be able to explain the "why" behind this transition and the trade-offs involved in various sequence modeling strategies.

---

# 1. 🔹 Sequence Modeling Evolution

## Q1: Why did Transformers replace RNNs/LSTMs?

### 🔹 Direct Answer
Transformers solved the **sequential bottleneck**. RNNs process tokens one-by-one, which prevents parallelization and makes it difficult to capture long-range dependencies due to vanishing gradients. Transformers use **Self-Attention**, allowing every token in a sequence to "attend" to every other token simultaneously, enabling massive parallelism and better long-term memory.

### 🔹 Intuition
Imagine reading a book.
- **RNN/LSTM:** You read word-by-word. By the end of a long sentence, you might forget how it started.
- **Transformer:** You look at the entire page at once. Your brain immediately connects "it" to the noun it refers to 10 lines above.

### 🔹 Deep Dive: Inductive Bias
RNNs have a strong **Recurrence Bias** (nearby items are more related). Transformers have a **Relational Bias** through attention, which is more flexible but requires significantly more data to learn the structure of language from scratch.

---

# 2. 🔹 Word Embeddings: The Bridge to Meaning

## Q2: Word2Vec vs. GloVe vs. FastText.

### 🔹 Comparison Table

| Feature | Word2Vec (2013) | GloVe (2014) | FastText (2016) |
| :--- | :--- | :--- | :--- |
| **Method** | Predictive (Skip-gram/CBOW). | Global Vector (Matrix Factorization). | Character n-grams. |
| **Strength** | Fast to train; captures analogies. | Leverages global co-occurrence stats. | Handles **OOV (Out-of-Vocabulary)** well. |
| **Weakness** | Ignores global context. | Memory intensive. | Slower to train (sub-word units). |

### 🔹 Deep Dive: FastText and OOV
Unlike Word2Vec, which maps each word to a vector, FastText breaks words into n-grams (e.g., "apple" -> "ap", "pp", "pl", "le"). This allows it to generate a representation for a word it has *never* seen before (like "apple-ish") by averaging its sub-word vectors.

---

# 3. 🔹 The Transformer Architecture

## Q3: Explain Scaled Dot-Product Attention.

### 🔹 Direct Answer
Attention computes a weighted sum of values based on the compatibility of a query with its keys:
$$Attention(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
The scaling factor $\sqrt{d_k}$ is critical: it prevents the values inside the softmax from becoming too large, which would lead to vanishing gradients in the early layers.

### 🔹 Intuition: The Library Analogy
1. **Query (Q):** Your search query (e.g., "Climate Change").
2. **Key (K):** The labels on the spines of books in the library.
3. **Value (V):** The actual content inside those books.
The model compares your search (Q) to all labels (K) and retrieves the most relevant content (V).

---

# 4. 🔹 Pre-training Paradigms

## Q4: BERT (Encoder) vs. GPT (Decoder).

### 🔹 Direct Answer
- **BERT (Encoder-only):** Trained via Masked Language Modeling (**MLM**). It is **bidirectional**, seeing both left and right context. Ideal for NLU tasks (Classification, NER, Entailment).
- **GPT (Decoder-only):** Trained via Causal Language Modeling (**CLM**). It is **unidirectional** (left-to-right). Ideal for generative tasks.

### 🔹 Practical Perspective: Fine-tuning
When performing **Sentiment Analysis**, BERT is superior because it can use context from the entire sentence to understand a word. When building a **Chatbot**, GPT is required because it must predict the "next token" sequentially.

---

# 5. 🔹 Tokenization Strategies

## Q5: Why is BPE preferred over Word or Character tokenization?

### 🔹 Direct Answer
**Byte-Pair Encoding (BPE)** is a sub-word tokenization method that finds the optimal balance:
1. **Word-level:** Leads to massive vocabularies and "Out of Vocabulary" (OOV) issues for rare words.
2. **Character-level:** Sequences become too long, making learning relationships difficult.
**BPE** breaks rare words into common sub-words (e.g., "Transformers" -> "Trans" + "formers"). This allows a fixed vocabulary (e.g., 50k tokens) to represent almost any word in the language.

---

# 6. 🔹 Practical Perspective (Code)

### **Multi-Head Attention (High-Yield Concept)**
```python
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def forward(self, q, k, v):
        # 1. Similarity Scores
        scores = (q @ k.transpose(-2, -1)) / (q.size(-1)**0.5)
        # 2. Weights
        weights = F.softmax(scores, dim=-1)
        # 3. Weighted Sum
        return weights @ v
```

---

## 🔹 Difficulty Tag: 🔴 Hard
