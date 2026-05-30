---
module: Specialized Domains
topic: Natural Language Processing
subtopic: ""
status: unread
tags: [nlp, word2vec, lstm, bert, gpt, transformers, specialized-domains]
---
# Natural Language Processing

---

## Table of Contents

1. [Why NLP is Hard](#1-why-nlp-is-hard)
2. [Pre-Neural NLP: Classical Approaches](#2-pre-neural-nlp-classical-approaches)
3. [Word Embeddings: Word2Vec, GloVe, FastText](#3-word-embeddings-word2vec-glove-fasttext)
4. [Recurrent Architectures: RNN, LSTM, GRU](#4-recurrent-architectures-rnn-lstm-gru)
5. [Seq2Seq and Attention](#5-seq2seq-and-attention)
6. [ELMo and Contextual Embeddings](#6-elmo-and-contextual-embeddings)
7. [The Transformer Architecture](#7-the-transformer-architecture)
8. [BERT and Encoder Models](#8-bert-and-encoder-models)
9. [GPT and Decoder Models](#9-gpt-and-decoder-models)
10. [T5 and Encoder-Decoder Models](#10-t5-and-encoder-decoder-models)
11. [Common Interview Questions](#11-common-interview-questions)

---

## 1. Why NLP is Hard

Language is discrete, high-dimensional, and compositional. Unlike images (where nearby pixels are correlated), words have no natural ordering in embedding space. The same word means different things in different contexts ("bank" as financial institution vs. riverbank). Understanding a sentence requires resolving long-range dependencies ("The trophy didn't fit in the suitcase because it was too big" — what is "it"?). Meaning is often implicit, ironic, cultural.

**Three fundamental challenges:**

1. **Polysemy:** Same token, multiple meanings. Context-free representations (word2vec) average over meanings.
2. **Long-range dependencies:** The subject of a verb can be 50 tokens away. RNNs struggle with long-range credit assignment; transformers solve this with direct attention.
3. **Compositionality:** Meaning of "not happy" is not the sum of "not" and "happy" — composition is non-trivial and context-dependent.

---

## 2. Pre-Neural NLP: Classical Approaches

### Bag of Words (BoW)

Represent a document as a vector of word counts. Discard word order entirely. Simple but loses all sequential information.

**TF-IDF (Term Frequency–Inverse Document Frequency):**
```
TF(t, d) = count(t in d) / |d|
IDF(t) = log(N / df(t))
TF-IDF(t, d) = TF(t, d) × IDF(t)
```

Down-weights common words (the, is) that appear in almost every document; up-weights rare words that characterize specific documents. Still the baseline for many document retrieval and classification tasks.

### N-gram Language Models

Estimate P(w_t | w_{t-1}, ..., w_{t-n+1}) — probability of next word given n-1 preceding words. Store empirical counts; apply Laplace smoothing or Kneser-Ney smoothing for unseen n-grams.

**What breaks:** Exponential growth in vocabulary: V^n possible n-grams. Sparsity: most n-grams never appear in training data. Generalization: trigram "the cat sat" and "the dog sat" share no evidence.

### Named Entity Recognition and POS Tagging

Traditional approach: features (word, prefix, suffix, capitalization, surrounding words) → CRF (Conditional Random Field) sequence model. CRF models P(y_1, ..., y_n | x_1, ..., x_n) globally, enforcing valid label sequences (I-PER cannot follow B-LOC).

---

## 3. Word Embeddings: Word2Vec, GloVe, FastText

**The problem:** One-hot vectors have dimension |V| (50,000+), are orthogonal (no semantic relationship), and require learning the meaning of every word from scratch in every model.

**The core insight (distributional hypothesis):** Words that occur in similar contexts have similar meanings. "Dog" and "cat" both appear near "pet," "food," "vet" — they should have similar representations.

### Word2Vec (Mikolov et al., 2013)

Two architectures, both learn dense d-dimensional vectors for every word:

**CBOW (Continuous Bag of Words):** Predict the center word from surrounding context words.
```
Input: context words (one-hot) → average → hidden layer → output: center word probability
```

**Skip-gram:** Predict surrounding context words from the center word.
```
Input: center word → hidden layer → output: context word probability (each window position separately)
```

**Skip-gram with Negative Sampling (SGNS):**
```python
# For each (center, context) pair:
# Positive: increase similarity(center, context)
# Negative: decrease similarity(center, k random words)

L = log σ(v_c · v_w) + Σ_{i=1}^{k} E_{w_neg ~ P_n}[log σ(-v_c · v_{w_neg})]
```

Negative sampling avoids the expensive full softmax over |V| words. Noise distribution P_n ∝ freq(w)^(3/4) — slight smoothing of word frequency.

**Famous property:** Linear algebraic relationships capture semantic analogies:
```
king - man + woman ≈ queen
Paris - France + Germany ≈ Berlin
```

This emerges from training — the model did not learn analogies explicitly.

**What breaks:** Single vector per word — no context. "bank" in financial context and riverbank context have the same embedding. This is the key limitation that ELMo and BERT fix.

### GloVe (Global Vectors, Pennington et al., 2014)

**The insight:** Word2Vec uses a local context window. But global co-occurrence statistics of the corpus contain signal — the ratio P(solid|ice)/P(solid|steam) is informative. GloVe trains on the full co-occurrence matrix.

**Objective:** Learn vectors such that their dot product approximates the log co-occurrence count:
```
J = Σ_{i,j} f(X_{ij}) (w_i · w_j + b_i + b_j - log X_{ij})²
```

f(X) is a weighting function that diminishes the contribution of very frequent pairs (to prevent "the" from dominating).

**In practice:** GloVe and Word2Vec perform similarly. GloVe trains faster on large corpora; Word2Vec captures slightly more local syntactic patterns.

### FastText (Bojanowski et al., 2017)

**The insight:** Word2Vec has no representation for out-of-vocabulary (OOV) words or morphological variants. "running" and "runs" share no information despite sharing the root "run."

**Key idea:** Represent each word as a bag of character n-grams. Word "where" → {<wh, whe, her, ere, re>, <where>}. The word vector is the sum of its n-gram vectors.

**What this buys:**
- OOV words: "unseen" → sum of its character n-grams — no full OOV
- Morphological generalization: "walk," "walking," "walked" share n-grams → share representation
- Better for morphologically rich languages (Finnish, Turkish, Arabic)

---

## 4. Recurrent Architectures: RNN, LSTM, GRU

**The problem:** Sentences are sequences of variable length. A fixed-size input representation loses the ordering information that is critical for meaning ("dog bites man" vs. "man bites dog").

### Vanilla RNN

**The core idea:** Maintain a hidden state h_t that summarizes all information seen up to position t. Update it at each step using the current input and previous hidden state.

```
h_t = tanh(W_hh h_{t-1} + W_xh x_t + b)
y_t = W_hy h_t + b_y
```

**What breaks — Vanishing gradients:** The gradient of the loss with respect to h_t propagates backward through time: ∂L/∂h_1 = ∂L/∂h_T × Π_{t=2}^{T} (∂h_t/∂h_{t-1}). Each factor is the Jacobian of tanh — eigenvalues typically < 1. Over 50+ steps, this product becomes exponentially small. The model cannot learn long-range dependencies: "the dog that [20 words] was happy" — the word "dog" determines "was" (singular), but the gradient carrying this information vanishes before reaching the verb.

### LSTM (Long Short-Term Memory, Hochreiter & Schmidhuber, 1997)

**The core insight:** Add a **cell state** c_t — a "conveyor belt" that carries information across many steps with minimal transformation. Gates control what information is added to, removed from, and read from the cell state.

**Four components:**

```
Forget gate:   f_t = σ(W_f [h_{t-1}, x_t] + b_f)
Input gate:    i_t = σ(W_i [h_{t-1}, x_t] + b_i)
Cell candidate: g_t = tanh(W_g [h_{t-1}, x_t] + b_g)
Output gate:   o_t = σ(W_o [h_{t-1}, x_t] + b_o)

Cell update:   c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
Hidden update: h_t = o_t ⊙ tanh(c_t)
```

**Why LSTM solves vanishing gradients:** The gradient through c_t is controlled by the forget gate. When f_t = 1 (keep everything), the gradient flows unchanged. The additive update (c_t = f × c_{t-1} + i × g) avoids the multiplicative chain that causes vanishing. In a standard RNN, every step multiplies by the same weight matrix; in LSTM, the cell update is additive.

**Intuition for each gate:**
- **Forget gate:** "Should I drop what I remembered?" — useful for switching contexts (end of one sentence, start of another)
- **Input gate:** "Is the new information worth adding to memory?"
- **Cell state:** Long-term memory — protected from gradient vanishing
- **Output gate:** "What should I expose from memory right now?"

```python
import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, bidirectional=True, dropout=0.3)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)  # ×2 for bidirectional

    def forward(self, x):
        emb = self.embedding(x)               # (batch, seq_len, embed_dim)
        out, (h_n, c_n) = self.lstm(emb)      # out: (batch, seq_len, hidden*2)
        # Use last hidden state from both directions
        h = torch.cat([h_n[-2], h_n[-1]], dim=1)  # (batch, hidden*2)
        return self.classifier(h)
```

### GRU (Gated Recurrent Unit, Cho et al., 2014)

Simplification of LSTM: merges cell and hidden state into one. Two gates instead of three.

```
Reset gate:  r_t = σ(W_r [h_{t-1}, x_t])
Update gate: z_t = σ(W_z [h_{t-1}, x_t])
Candidate:   h̃_t = tanh(W [r_t ⊙ h_{t-1}, x_t])
Hidden:      h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t
```

**Update gate:** Controls how much of the previous hidden state to keep vs. how much of the new candidate to use — combining forget and input gates from LSTM.

**In practice:** GRU and LSTM perform similarly. GRU trains ~33% faster (fewer parameters). LSTM is more expressive and preferred for very long sequences or tasks requiring fine-grained memory control.

### Bidirectional RNNs

**Insight:** In most NLP tasks, context from both directions helps. "I went to the [bank] to deposit money" — "deposit money" after "bank" disambiguates it. A forward-only RNN at "bank" has only seen left context.

**BiLSTM:** Run one LSTM forward (left-to-right) and one backward (right-to-left). Concatenate hidden states at each position: h_t = [→h_t; ←h_t]. Each position has access to the full sentence context.

---

## 5. Seq2Seq and Attention

**The problem:** Tasks like machine translation, summarization, and question answering require mapping a variable-length input to a variable-length output. A single fixed-size vector (the encoder's final hidden state) cannot represent long, complex input sequences — the "information bottleneck."

### Sequence-to-Sequence (Sutskever et al., 2014)

**Architecture:** Encoder LSTM processes the source sentence and produces a context vector c (final hidden state). Decoder LSTM generates the target sequence one token at a time, conditioned on c.

```
Encoder: h_1, h_2, ..., h_T → c = h_T
Decoder: P(y_1, ..., y_T') = Π P(y_t | y_{<t}, c)
```

**What breaks:** The fixed-size context vector c must encode the entire input sequence. For long sentences (20+ words), critical information is compressed beyond what the vector can retain. Translation quality degrades significantly for long sentences.

### Attention (Bahdanau et al., 2015)

**The core insight:** Instead of compressing the input to a single vector, allow the decoder at each step to directly attend to all encoder hidden states, weighted by relevance.

**Mechanism:**
```
At decoder step t:
1. Compute alignment scores: e_{t,i} = a(s_{t-1}, h_i)  (s = decoder state, h = encoder states)
   Common choice: e_{t,i} = v^T tanh(W_s s_{t-1} + W_h h_i)

2. Normalize: α_{t,i} = softmax(e_{t,i}) = exp(e_{t,i}) / Σ_j exp(e_{t,j})

3. Context vector: c_t = Σ_i α_{t,i} h_i  (weighted sum of encoder states)

4. Decode: s_t = f(s_{t-1}, y_{t-1}, c_t)
```

**What this buys:** The decoder can "look back" at any part of the input at each step. For translating "The cat sat on the mat" → "Le chat s'est assis sur le tapis," when decoding "chat," attention peaks on "cat." Interpretable alignment weights.

**What breaks:** O(T × T') attention computations — each decoder step attends to all encoder steps. Still sequential — cannot parallelize across encoder or decoder steps. This motivated the Transformer.

---

## 6. ELMo and Contextual Embeddings

**The problem:** Word2Vec gives the same embedding for "bank" in all contexts. Real language understanding requires context-sensitive representations.

**ELMo (Embeddings from Language Models, Peters et al., 2018):**

**The core insight:** Train a bidirectional language model (biLM) on large text. Use the internal representations of the biLM as word embeddings. Because the representations depend on the full sentence, "bank" in different contexts has different embeddings.

**Architecture:**
- Two BiLSTM layers trained on language modeling (predict next word forward, predict previous word backward)
- ELMo embedding = weighted sum of all BiLSTM layer representations (weights are task-specific, learned during fine-tuning)

```
ELMo(w_k) = γ^task Σ_j s_j^task h_{k,j}^{LM}
```

s_j = softmax-normalized task-specific scalar weights. h_{k,j} = hidden state at layer j for token k.

**What this buys:** ELMo gave large improvements across NLP tasks by providing contextual word representations. First demonstration that pre-trained language model representations transfer broadly.

**What breaks:** Still RNN-based — cannot parallelize. Representations are fixed after pre-training; fine-tuning updates only the task layer, not the ELMo parameters (in the original paper). BERT later introduced full fine-tuning of the pre-trained model, which works better.

---

## 7. The Transformer Architecture

**The problem:** RNNs process sequences token-by-token, which (1) prevents parallelization during training — step t requires step t-1's hidden state, and (2) makes long-range dependencies hard — information from position 1 must propagate through all intermediate states to reach position T.

**The core insight (Vaswani et al., 2017 — "Attention Is All You Need"):** Replace recurrence entirely with self-attention. Every token attends directly to every other token in a single operation. No sequential processing — all positions computed in parallel. Long-range dependencies are free.

### Self-Attention

**The mechanism:** Each token produces a Query, Key, and Value vector via learned projections. Attention weights are computed by comparing queries against keys; the output is a weighted sum of values.

```
Q = X W_Q,  K = X W_K,  V = X W_V

Attention(Q, K, V) = softmax(Q K^T / √d_k) V
```

**Why √d_k:** Without scaling, dot products grow large when d_k is large, pushing softmax into saturation (near-zero gradients). Scaling by √d_k keeps magnitudes stable.

**What each piece does:**
- **Query:** "What am I looking for?"
- **Key:** "What do I contain?"
- **Value:** "What should I pass along if attended to?"

Each token's output is a weighted combination of all tokens' values — weighted by how compatible each token's key is with the current token's query.

### Multi-Head Attention

Run h parallel attention heads with different projection matrices:
```python
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W_O

head_i = Attention(Q W_Qi, K W_Ki, V W_Vi)
```

Each head learns to attend to different aspects — one head might track syntactic dependencies; another semantic similarity; another coreference.

```python
import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

    def split_heads(self, x):
        B, T, _ = x.size()
        return x.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        # → (B, heads, T, d_k)

    def forward(self, Q, K, V, mask=None):
        Q, K, V = self.split_heads(self.W_Q(Q)), self.split_heads(self.W_K(K)), self.split_heads(self.W_V(V))
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = scores.softmax(dim=-1)
        out = (attn @ V).transpose(1, 2).contiguous().view(Q.size(0), -1, self.num_heads * self.d_k)
        return self.W_O(out)
```

### Positional Encoding

Self-attention is permutation-equivariant — it treats all tokens identically regardless of order. To inject position information, add a positional encoding to each token embedding.

**Sinusoidal encoding (original paper):**
```
PE(pos, 2i)   = sin(pos / 10000^{2i/d_model})
PE(pos, 2i+1) = cos(pos / 10000^{2i/d_model})
```

Different frequencies for different dimensions. Properties: each position has a unique encoding; the encoding for position p+k can be expressed as a linear function of position p — the model can attend by relative position.

**Learned positional embeddings (BERT, GPT):** Train a position embedding table — one vector per position. More flexible; limited to the maximum sequence length seen at training time.

**Rotary Positional Embedding (RoPE, used in LLaMA, GPT-NeoX):** Encode relative positions by rotating query and key vectors. The dot product Q·K^T naturally produces a function of the relative position — no absolute positional information injected.

### Transformer Block

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Pre-norm (modern) or Post-norm (original paper)
        x = x + self.dropout(self.attention(self.norm1(x), self.norm1(x), self.norm1(x), mask))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x
```

**Feed-forward network:** Applied position-wise (same MLP at every position). d_ff = 4 × d_model typically. Projects to a higher-dimensional space where the nonlinearity (GELU) can perform more computation, then projects back.

**Layer Norm vs. Batch Norm for NLP:** BatchNorm normalizes across the batch dimension — requires a consistent batch size and doesn't handle variable-length sequences well. LayerNorm normalizes across the feature dimension for each token independently — works with any sequence length, any batch size.

---

## 8. BERT and Encoder Models

**The problem:** ELMo used BiLSTMs to get contextual representations. Transformers process all positions in parallel. Can we combine transformer architecture with bidirectional pre-training to get better, parallelizable contextual representations?

**BERT (Bidirectional Encoder Representations from Transformers, Devlin et al., 2018):**

**Architecture:** Encoder-only transformer. 12 layers (BERT-base) or 24 layers (BERT-large). Bidirectional — each token attends to all other tokens in both directions in every layer.

### Pre-Training Tasks

**Masked Language Modeling (MLM):**
- Randomly mask 15% of input tokens
- Of masked tokens: 80% replaced with [MASK], 10% replaced with random word, 10% left unchanged
- Train to predict the original token at masked positions

Why not 100% [MASK]?: At fine-tuning time, there are no [MASK] tokens. The 10%/10% mixture closes the train-test distribution gap — the model learns not to rely on the [MASK] token identity.

**Next Sentence Prediction (NSP):**
- Given two sentences A and B: 50% of the time B actually follows A; 50% it's a random sentence
- Train a binary classifier: is B the actual next sentence?

NSP was later found to be of limited value (RoBERTa removed it with improved performance).

### Fine-Tuning

```
[CLS] token 1 token 2 ... [SEP] token N [SEP]
```

For classification: take the CLS token representation, add a linear head.
For token-level tasks (NER, QA): take each token's representation, add a linear head per token.

```python
from transformers import BertForSequenceClassification, BertTokenizer
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

inputs = tokenizer("I love this movie", return_tensors='pt', padding=True, truncation=True)
outputs = model(**inputs)
logits = outputs.logits
```

### RoBERTa and BERT Improvements

**RoBERTa (Liu et al., 2019):**
- Train longer with larger batch size
- More data (160GB vs. 16GB)
- Remove NSP task — hurts performance
- Dynamic masking: new mask pattern generated each epoch rather than static mask

**ALBERT:** Factorize embedding matrix (vocab_size × d_model → vocab_size × d_emb + d_emb × d_model), cross-layer parameter sharing. Smaller model, same performance.

**DistilBERT:** Knowledge distillation from BERT. 60% of parameters, 97% of performance, 60% faster inference.

### When to Use BERT-Style Models

Best for tasks that require understanding the full context of each token:
- Text classification (sentiment, topic)
- Named entity recognition
- Question answering (extractive: find span in passage)
- Semantic similarity, natural language inference

**Not suited for:** Generation tasks (BERT is encoder-only; cannot generate autogressively).

---

## 9. GPT and Decoder Models

**The core insight:** Language modeling — predict the next token — is a powerful self-supervised objective that scales well. Unlike BERT's masked LM, causal LM is naturally generative.

**GPT (Generative Pre-trained Transformer, Radford et al., 2018):**

**Architecture:** Decoder-only transformer with causal (left-to-right) attention mask. Token at position t can only attend to positions ≤ t. This enforces the autoregressive generation property.

```
Causal mask: upper triangular matrix of -inf
Applied before softmax in attention, so future tokens contribute zero
```

### Causal Attention Mask

```python
def causal_mask(seq_len):
    mask = torch.tril(torch.ones(seq_len, seq_len))
    return mask  # 0 in upper triangle → -inf before softmax
```

### GPT-2 and Scaling

GPT-2 (2019): 1.5B parameters. Trained on WebText (40GB). Demonstrated that large LMs perform surprisingly well on many tasks in a zero-shot setting — prompted via task description in natural language.

**Few-shot prompting:** Provide a few examples of the task format in the prompt context. The model "in-context learns" without any weight updates:
```
Translate English to French:
sea otter → loutre de mer
peppermint → menthe poivrée
plush girafe → girafe peluche
cheese →
```

### GPT-3 and In-Context Learning

GPT-3 (2020): 175B parameters. Key insight: with sufficient scale, the model can perform few-shot learning via in-context examples without any fine-tuning. Performance scales predictably with model size, dataset size, and compute (scaling laws — Hoffmann et al., Chinchilla paper).

**Emergent abilities:** Some capabilities appear only above a certain parameter threshold — below that threshold, performance is near-random; above it, the model suddenly demonstrates the ability. Examples: multi-step arithmetic, chain-of-thought reasoning.

### Instruction Tuning and RLHF

**The problem:** A base language model is optimized to predict text on the internet, not to be helpful, harmless, or follow instructions. Prompting tricks are brittle and inconsistent.

**InstructGPT / ChatGPT pipeline:**
1. **Supervised Fine-Tuning (SFT):** Fine-tune the base model on human-written demonstrations of desired behavior
2. **Reward Model:** Collect human preference labels on model outputs; train a scalar reward model
3. **PPO:** Fine-tune the SFT model with PPO using the reward model signal, with a KL penalty vs. SFT model

**DPO (Direct Preference Optimization):** Skip the explicit reward model. Reparameterize the RL objective to optimize directly from preference pairs (preferred, rejected). See RLHF coverage in the RL section.

---

## 10. T5 and Encoder-Decoder Models

**The core insight (T5 — Text-to-Text Transfer Transformer, Raffel et al., 2020):** Frame every NLP task as text-to-text. Classification → "label: positive." Translation → target sentence. QA → answer string. This allows one model architecture and one loss (cross-entropy) for all tasks.

```
Task-specific prompts at input:
"translate English to German: The house is wonderful."
"summarize: The tower is 324 metres ..."
"cola sentence: The cat sat on the mat."
```

**Architecture:** Standard encoder-decoder transformer. Encoder processes the input bidirectionally; decoder generates the output autoregressively.

**Why encoder-decoder for generation?** The encoder can see the full input bidirectionally (unlike a decoder-only model constrained to causal attention). The decoder generates output attending to all encoder states via cross-attention. Best for tasks with a distinct input-output structure (translation, summarization, data-to-text).

### BART

Trained with a denoising objective: corrupt the input (masking, shuffling, deletion, rotation), train the model to reconstruct the original. Acts as a generalization of BERT (encoder) and GPT (decoder).

Strong performance on summarization (CNN/DailyMail), translation, abstractive QA.

### Encoder vs. Decoder vs. Encoder-Decoder

| Architecture | Examples | Best for |
|---|---|---|
| Encoder-only | BERT, RoBERTa, DeBERTa | Classification, NER, extractive QA |
| Decoder-only | GPT-2/3/4, LLaMA, Mistral | Text generation, in-context learning |
| Encoder-Decoder | T5, BART, mT5 | Translation, summarization, seq2seq tasks |

---

## 11. Common Interview Questions

### Q1: What is the vanishing gradient problem in RNNs? How does LSTM fix it?

**Vanishing gradient:** In backpropagation through time (BPTT), gradients multiply the Jacobian of tanh at each step. The tanh Jacobian has maximum eigenvalue < 1; over 50+ steps, this product → 0. Early tokens receive no gradient — the model cannot learn long-range dependencies.

**LSTM fix:** The cell state c_t is updated additively: c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t. When f_t ≈ 1 (forget gate open), the gradient ∂c_t/∂c_{t-1} ≈ 1 — no vanishing. The forget gate learns to preserve important information across many steps.

---

### Q2: Why does the Transformer use √d_k scaling in attention?

Without scaling, for d_k-dimensional query and key vectors with unit variance, the dot product Q·K has variance d_k. Large dot products push softmax into regions where gradients are exponentially small (saturation). Scaling by 1/√d_k normalizes the dot product to unit variance, keeping softmax in a well-conditioned regime.

---

### Q3: What is the difference between BERT and GPT pre-training?

**BERT (encoder-only):** Masked language modeling — randomly mask tokens, train to reconstruct them. Bidirectional — every token attends to every other token. Not autoregressive; cannot generate text. Strong at understanding/classification.

**GPT (decoder-only):** Causal language modeling — predict next token from left context only. Unidirectional — each token attends only to previous tokens. Autoregressive; naturally generates text. Strong at generation, in-context learning.

**Key tradeoff:** Bidirectional attention (BERT) gives richer representations for each token but breaks the autoregressive generation property. Causal attention (GPT) enables generation but each token's representation lacks right-side context.

---

### Q4: What is in-context learning? How does it differ from fine-tuning?

**In-context learning:** Provide task examples in the prompt; the model generates outputs that follow the pattern, without any weight updates. The model "adapts" via attention over the context — not via gradient descent.

**Fine-tuning:** Update model weights on task-specific labeled data via gradient descent.

**Key differences:**
- In-context: no weight update, no labeled data needed (just examples in prompt), expensive per-inference (long context), performance depends on prompt format, limited to tasks representable as text
- Fine-tuning: weight update required, needs labeled data, cheaper inference (shorter input), generally higher performance for specialized tasks, harder to update for new tasks

---

### Q5: What is the attention complexity problem? What solutions exist?

Standard self-attention: O(n²) in sequence length and memory. For n=4096 tokens, the attention matrix is 4096×4096 = 16M entries per head.

**Solutions:**
- **Sparse attention (Longformer, BigBird):** Each token attends to local window + a few global tokens + random tokens. O(n) complexity.
- **Linear attention:** Approximate softmax attention with a kernel trick to decompose Q(K^T V) → Q(K^T)(V). O(n).
- **Sliding window + global tokens (Longformer):** Local window (512 tokens) for most tokens; global tokens for [CLS] and task-specific tokens that need full context.
- **Flash Attention (Dao et al., 2022):** Same O(n²) complexity but dramatically reduced IO operations by computing attention in tiles that fit in GPU SRAM. Not asymptotically better; 2–4× faster in practice and exact (not approximate).
- **Grouped query attention (GQA):** Multiple query heads share a single key/value head. Reduces KV cache memory in inference by h_kv/h_q ratio.

---

### Q6: What is word2vec's skip-gram negative sampling objective?

For each (center_word, context_word) positive pair, predict that the context word appears (positive example) while predicting that k randomly sampled words do not appear (negative examples):

```
L = log σ(v_c · v_w) + Σ_{i=1}^{k} E_{w_neg}[log σ(-v_c · v_{w_neg})]
```

This avoids computing the full softmax over the entire vocabulary (expensive) by instead solving a series of binary classifications. Negative samples drawn from P_n(w) ∝ freq(w)^(3/4) — slight frequency smoothing reduces over-representation of very common words.

---

### Q7: Why does ELMo use a weighted sum of layer representations?

Different layers in a deep BiLSTM capture different aspects of language:
- Lower layers: syntax, morphology, part-of-speech
- Upper layers: semantics, word sense disambiguation

Different downstream tasks benefit from different layers. Named entity recognition benefits more from lower layers (morphology matters). Coreference resolution benefits more from upper layers (semantics matters).

ELMo's task-specific scalar weights (softmax-normalized) allow each downstream task to learn its own blend of layer representations. These weights are learned during fine-tuning.

---

## Quick Reference: NLP Architecture Timeline

| Era | Key Models | Pre-training | Representation |
|---|---|---|---|
| Pre-neural | TF-IDF, n-grams, CRF | None | Sparse bag-of-words |
| Static embeddings | Word2Vec, GloVe, FastText | Next-word prediction | Dense, context-free |
| Contextual RNN | ELMo | Bidirectional LM | Context-sensitive, sequential |
| Transformer era | BERT, GPT, T5 | MLM / causal LM | Context-sensitive, parallel |
| Large LLM era | GPT-3/4, LLaMA, Claude | Causal LM at scale | Emergent in-context learning |

---

## Key Papers

| Paper | Year | Contribution |
|---|---|---|
| Mikolov et al. | 2013 | Word2Vec — skip-gram, CBOW, negative sampling |
| Pennington et al. | 2014 | GloVe — global co-occurrence matrix factorization |
| Sutskever et al. | 2014 | Seq2Seq — LSTM encoder-decoder |
| Bahdanau et al. | 2015 | Neural attention mechanism for seq2seq |
| Hochreiter & Schmidhuber | 1997 | LSTM — gated recurrent memory |
| Peters et al. | 2018 | ELMo — contextual BiLSTM representations |
| Vaswani et al. | 2017 | Transformer — attention is all you need |
| Devlin et al. | 2018 | BERT — bidirectional transformer pre-training |
| Radford et al. | 2018 | GPT — decoder-only generative pre-training |
| Raffel et al. | 2020 | T5 — text-to-text transfer transformer |
| Brown et al. | 2020 | GPT-3 — few-shot in-context learning |
| Liu et al. | 2019 | RoBERTa — robust BERT training |
| Dao et al. | 2022 | Flash Attention — IO-efficient exact attention |
