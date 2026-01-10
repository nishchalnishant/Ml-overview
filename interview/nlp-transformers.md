# NLP & Transformers: Interview Deep Dive

## 📋 Quick Reference
| Concept | Definition | Interview Key |
|---------|------------|---------------|
| **Tokenization** | Split text into tokens | BPE, WordPiece, SentencePiece |
| **Embedding** | Token → Dense vector | Word2Vec, GloVe, Learned |
| **Attention** | Weigh all tokens | $Softmax(QK^T/\sqrt{d})V$ |
| **BERT** | Encoder-only, bidirectional | MLM, NSP, fine-tuning |
| **GPT** | Decoder-only, autoregressive | Next-token prediction |

---

## 🔤 1. Text Preprocessing

### Tokenization Methods
- **Word-level**: Simple but huge vocabulary. OOV problem.
- **Character-level**: Small vocabulary but loses word semantics.
- **Subword (BPE)**: Best of both. "unhappiness" → ["un", "happiness"]. Handles OOV.

### Common Preprocessing Steps
1. Lowercasing (context-dependent)
2. Removing special characters/URLs
3. Handling contractions ("don't" → "do not")
4. Lemmatization/Stemming (less common with modern LLMs)

---

## 🧠 2. Word Representations

### Traditional (Pre-Transformer Era)
- **TF-IDF**: Sparse, based on word frequency. Good for simple retrieval.
- **Word2Vec**: Dense embeddings learned via Skip-gram or CBOW.
- **GloVe**: Learned from co-occurrence matrix.

### Contextual (Modern)
- **ELMo**: First contextual embeddings (BiLSTM).
- **BERT/GPT**: Context-dependent from Transformers. "Bank" in "river bank" vs "money bank" gets different vectors.

### Interview Q: "Why are contextual embeddings better?"
> Static embeddings (Word2Vec) give the same vector to "bank" regardless of context. Contextual embeddings (BERT) produce different vectors based on surrounding words, capturing polysemy and nuance.

---

## 🤖 3. Transformer Variants

### BERT (Bidirectional Encoder Representations from Transformers)
- **Architecture**: Encoder-only.
- **Pre-training**: 
  - **MLM (Masked Language Modeling)**: Predict [MASK]ed words.
  - **NSP (Next Sentence Prediction)**: Is sentence B the next sentence after A?
- **Fine-tuning**: Add a classification head for downstream tasks.

### GPT (Generative Pre-trained Transformer)
- **Architecture**: Decoder-only.
- **Pre-training**: Next-token prediction (Causal Language Modeling).
- **Use**: Generation tasks (chatbots, code generation).

### T5 (Text-to-Text Transfer Transformer)
- **Architecture**: Full Encoder-Decoder.
- **Paradigm**: Every task is framed as "text-in, text-out".

### Interview Q: "When to use BERT vs GPT?"
> **BERT** for understanding/classification (sentiment, NER, Q&A). **GPT** for generation (chatbots, summarization, creative writing). If you need both, use T5 or an Encoder-Decoder.

---

## 📐 4. Attention Mechanisms

### Self-Attention
Every token attends to every other token in the sequence.
$$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

### Multi-Head Attention
Run attention $h$ times with different projections, then concatenate.
- Allows model to focus on different relationships simultaneously.

### Cross-Attention
Used in Encoder-Decoder models. Query from Decoder, Key/Value from Encoder.

### Interview Q: "What is Causal (Masked) Attention?"
> In GPT, we mask future tokens so the model can only "see" past. The attention matrix is upper-triangular (or lower, depending on convention). This is what makes it autoregressive.

---

## 📊 5. Common NLP Tasks & Approaches

| Task | Type | Approach |
|------|------|----------|
| **Sentiment Analysis** | Classification | Fine-tune BERT with [CLS] token |
| **Named Entity Recognition (NER)** | Token Classification | Predict label per token |
| **Question Answering** | Extractive | Predict start/end span indices |
| **Summarization** | Generation | Encoder-Decoder (T5, BART) |
| **Translation** | Seq2Seq | Encoder-Decoder |

---

## ❓ Interview Questions

**"What is the difference between Encoder and Decoder in Transformers?"**
> The **Encoder** sees the whole input at once (bidirectional attention). The **Decoder** generates output autoregressively, seeing only past tokens (causal/masked attention). Encoder is for understanding, Decoder is for generation.

**"Why is positional encoding needed?"**
> Attention is permutation-invariant—it doesn't know token order. Positional encodings (sinusoidal or learned) are added to embeddings to inject sequence position information.

**"What is the difference between Fine-tuning and Feature Extraction?"**
> **Fine-tuning**: Update all weights of the pre-trained model on your task. **Feature Extraction**: Freeze the pre-trained model, only train a classifier head on top. Feature extraction is faster but less accurate; fine-tuning adapts the whole model to your domain.

**"How do you handle long documents that exceed the context window?"**
> Options: 1) Truncate (simple but lossy). 2) Sliding window with aggregation. 3) Hierarchical models (encode chunks, then aggregate). 4) Use long-context models (Longformer, BigBird) with sparse attention.
