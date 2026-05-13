# Deep Learning for NLP

NLP changed dramatically once deep learning stopped treating text like a bag of disconnected words and started treating it like structured context. That single shift — from counting to representing — changed everything.

---

## 1. The Evolution of NLP

| Era | Method | Key Idea | Limitation |
| :--- | :--- | :--- | :--- |
| **Pre-deep learning** | Bag-of-Words, TF-IDF | Word counts + frequency weighting | No word order, no semantics |
| **Static embeddings** | Word2Vec, GloVe, FastText | Dense word vectors from co-occurrence | One vector per word regardless of context |
| **Sequence models** | RNN, LSTM, GRU | Process tokens sequentially with hidden state | Slow to train, struggles with long-range dependencies |
| **Transformers** | BERT, GPT, T5, etc. | Attention over all tokens in parallel | $O(N^2)$ attention cost; long context is expensive |
| **Modern LLMs** | GPT-4, LLaMA, Mistral | Scale + RLHF + instruction tuning | Hallucination, compute cost, alignment |

---

## 2. Bag-of-Words and TF-IDF

**BoW** represents a document as a sparse vector of word counts.

$$\text{BoW}(d) \in \mathbb{R}^{|V|}$$

Loses word order entirely. "Dog bites man" = "Man bites dog."

**TF-IDF** weights words by how distinctive they are across the corpus:

$$\text{TF-IDF}(t, d) = \underbrace{\frac{f_{t,d}}{\sum_{t'} f_{t',d}}}_{\text{TF}} \times \underbrace{\log \frac{N}{|\{d : t \in d\}|}}_{\text{IDF}}$$

- TF: how often the term appears in this document
- IDF: log of inverse document frequency — penalizes common words like "the"

Still useful for:
- baseline text classifiers
- keyword extraction
- document similarity at scale

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=50000, ngram_range=(1, 2), sublinear_tf=True)),
    ('clf', LogisticRegression(C=1.0, max_iter=1000))
])
pipe.fit(X_train, y_train)
```

---

## 3. Word Embeddings

### Word2Vec

Word2Vec learns dense vector representations by training a shallow neural network on two proxy tasks:

**CBOW (Continuous Bag of Words):** predict center word from context.

**Skip-gram:** predict context words from center word.

$$J = -\frac{1}{T} \sum_{t=1}^{T} \sum_{-c \leq j \leq c, j \neq 0} \log P(w_{t+j} \mid w_t)$$

The key insight: words that appear in similar contexts have similar vectors. As a result:

$$\text{king} - \text{man} + \text{woman} \approx \text{queen}$$

**GloVe** trains directly on the global co-occurrence matrix, combining count-based and prediction-based methods:

$$J = \sum_{i,j} f(X_{ij}) \left( w_i^T \tilde{w}_j + b_i + \tilde{b}_j - \log X_{ij} \right)^2$$

**FastText** extends Word2Vec by representing words as sums of character n-grams. Handles:
- out-of-vocabulary (OOV) words
- morphologically rich languages
- misspellings

### Limitation of static embeddings

"bank" has one vector — the financial institution and the river bank are conflated. This is exactly the problem contextual embeddings (BERT) solve.

```python
import gensim.downloader as api

model = api.load("word2vec-google-news-300")
similarity = model.similarity('king', 'queen')
analogy = model.most_similar(positive=['king', 'woman'], negative=['man'])
```

---

## 4. Sequence Models (RNN, LSTM, GRU)

### Vanilla RNN

$$h_t = \tanh(W_h h_{t-1} + W_x x_t + b)$$

The hidden state $h_t$ carries information from previous time steps. Problem: gradients vanish or explode over long sequences because the same weight matrix $W_h$ is applied at every step.

### LSTM (Long Short-Term Memory)

Adds a **cell state** $c_t$ that acts as long-term memory, with three learned gates:

| Gate | Formula | Role |
| :--- | :--- | :--- |
| **Forget** | $f_t = \sigma(W_f [h_{t-1}, x_t] + b_f)$ | How much of $c_{t-1}$ to keep |
| **Input** | $i_t = \sigma(W_i [h_{t-1}, x_t] + b_i)$ | How much of new candidate to write |
| **Output** | $o_t = \sigma(W_o [h_{t-1}, x_t] + b_o)$ | How much of $c_t$ to expose |

$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$
$$h_t = o_t \odot \tanh(c_t)$$

Gates use sigmoid ($\in [0,1]$), so they act as soft switches. This enables gradients to flow through time without the multiplicative collapse of vanilla RNNs.

### GRU (Gated Recurrent Unit)

Simpler than LSTM — merges cell and hidden state into one, uses only reset and update gates:

$$z_t = \sigma(W_z [h_{t-1}, x_t])$$
$$r_t = \sigma(W_r [h_{t-1}, x_t])$$
$$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$

GRU matches LSTM performance on many tasks with fewer parameters.

### Why They Lost to Transformers

| Problem | Impact |
| :--- | :--- |
| Sequential processing | Cannot be parallelized across time steps — slow on GPUs |
| Long-range dependencies | Even LSTM degrades on very long sequences |
| Fixed-size bottleneck | The hidden state is a single vector summarizing all history |

---

## 5. Encoder-Decoder and Attention (Pre-Transformer)

For seq2seq tasks (translation, summarization), the encoder compresses input into a context vector, the decoder generates output.

**Bahdanau attention (2015)** fixes the bottleneck: instead of one context vector, the decoder attends over all encoder hidden states at each decoding step:

$$e_{ij} = a(s_{i-1}, h_j) \quad \text{(alignment score)}$$
$$\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_k \exp(e_{ik})} \quad \text{(attention weight)}$$
$$c_i = \sum_j \alpha_{ij} h_j \quad \text{(context vector)}$$

This was the direct predecessor to the self-attention in Transformers.

---

## 6. Transformers in NLP

Full architecture covered in [transformers.md](../components/transformers.md) and [attention.md](../components/attention.md). The key NLP-specific points:

### BERT (Bidirectional Encoder Representations from Transformers)

Encoder-only. Pre-trained with two tasks:
1. **Masked Language Modeling (MLM):** predict 15% of randomly masked tokens
2. **Next Sentence Prediction (NSP):** classify if two sentences are consecutive (later shown less important)

BERT is bidirectional — every token attends to all other tokens. This makes it excellent for:
- classification
- named entity recognition
- question answering
- sentence embeddings

**Fine-tuning:** add a task-specific head on top of the `[CLS]` token, fine-tune on labeled data.

```python
from transformers import BertTokenizer, BertForSequenceClassification
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

inputs = tokenizer("I loved this movie", return_tensors='pt', padding=True, truncation=True)
outputs = model(**inputs)
logits = outputs.logits
```

### GPT (Generative Pre-trained Transformer)

Decoder-only. Left-to-right causal language modeling: predict the next token given all previous tokens.

$$L = -\sum_t \log P(w_t \mid w_{<t})$$

Because it is causal (masked self-attention), each token only attends to past tokens. This makes GPT natural for generation tasks.

**GPT-3 and beyond:** in-context learning — the model learns from examples in the prompt without gradient updates.

### T5 (Text-to-Text Transfer Transformer)

Encoder-decoder. Frames every NLP task as text-to-text:
- Translation: `translate English to French: The cat sat on the mat`
- Summarization: `summarize: [document]`
- Classification: `sentiment: I loved this film` → `positive`

One unified training objective for all tasks.

### Key architectural variants

| Model family | Architecture | Primary use |
| :--- | :--- | :--- |
| **BERT, RoBERTa, DeBERTa** | Encoder-only | Classification, NER, QA |
| **GPT, LLaMA, Mistral** | Decoder-only | Generation, instruction following |
| **T5, BART, mT5** | Encoder-decoder | Translation, summarization |

---

## 7. Tokenization

Tokenization is underrated. The choice of tokenizer affects:
- vocabulary size (model parameter count)
- OOV handling
- multilingual coverage
- how numbers, code, and rare words are handled

### Byte-Pair Encoding (BPE)

Most common in modern LLMs (GPT-2, GPT-4, LLaMA).

**Algorithm:**
1. Start with character vocabulary
2. Repeatedly merge the most frequent adjacent pair into a new token
3. Repeat until vocabulary size is reached

"unbelievable" might tokenize as: `un`, `believ`, `able` — reuses common subword units.

### WordPiece (BERT)

Like BPE but merges based on likelihood gain rather than raw frequency. Subwords after the first start with `##`:
`"unbelievable"` → `un`, `##believ`, `##able`

### SentencePiece

Language-agnostic, operates on raw characters (no whitespace assumption). Used for multilingual models (mT5, XLM-R).

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokens = tokenizer.encode("unbelievable")
print(tokenizer.convert_ids_to_tokens(tokens))
# ['un', 'believ', 'able']
```

---

## 8. Text Classification Pipeline

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, Dataset
import torch

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.encodings = tokenizer(texts, truncation=True, padding=True,
                                   max_length=max_len, return_tensors='pt')
        self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.encodings.items()}, self.labels[idx]

tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

dataset = TextDataset(texts_train, labels_train, tokenizer)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

for epoch in range(3):
    for batch, labels in loader:
        outputs = model(**batch, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

---

## 9. Common NLP Tasks and Approaches

| Task | Approach | Metric |
| :--- | :--- | :--- |
| **Sentiment analysis** | Fine-tuned BERT / DistilBERT | F1, Accuracy |
| **Named entity recognition** | Token classification head on BERT | F1 per entity type |
| **Question answering** | Span extraction (BERT on SQuAD) | Exact Match, F1 |
| **Machine translation** | Fine-tuned T5 or NLLB | BLEU, chrF |
| **Summarization** | Fine-tuned BART or Pegasus | ROUGE-1/2/L |
| **Text generation** | Decoder-only LLM + sampling | Perplexity, human eval |
| **Semantic search** | Bi-encoder (Sentence-BERT) + ANN | MRR@K, nDCG@K |
| **Semantic similarity** | Cross-encoder (slower, more accurate) | Pearson r on STS-B |

---

## 10. Evaluation Metrics

**BLEU (translation):** modified n-gram precision with brevity penalty. Correlates with human judgment at corpus level but unreliable at sentence level.

**ROUGE (summarization):**
- ROUGE-N: n-gram recall between hypothesis and reference
- ROUGE-L: longest common subsequence

**Perplexity (language modeling):**

$$\text{PPL} = \exp\left(-\frac{1}{N} \sum_i \log P(w_i \mid w_{<i})\right)$$

Lower is better. A model with perplexity 50 is as uncertain as if it chose uniformly over 50 options at each token.

**BERTScore:** computes token-level similarity using contextual BERT embeddings. Correlates better with human judgment than BLEU/ROUGE.

---

## 11. Production Considerations

| Challenge | Solution |
| :--- | :--- |
| **Model size** | DistilBERT (6 layers), quantization (INT8), pruning |
| **Latency** | ONNX export, TorchScript, batch inference |
| **Multilingual** | mBERT, XLM-R, or language-specific models |
| **Domain shift** | Domain-adaptive pretraining (DAPT) on in-domain unlabeled text |
| **Class imbalance** | Focal loss, oversampling, threshold tuning |
| **Long documents** | Chunking + sliding window, Longformer, BigBird |

> [!TIP]
> **Interview structure:** For any NLP question — explain (1) representation: how you convert text to vectors, (2) model: which architecture and why, (3) evaluation: what metric and why it matters, (4) production: what breaks at scale.
