# Deep Learning for NLP

---

## Table of Contents

1. The Evolution of NLP
2. Bag-of-Words and TF-IDF
3. Word Embeddings
4. Sequence Models (RNN, LSTM, GRU)
5. Encoder-Decoder and Attention
6. Transformers in NLP
7. Tokenization
8. Text Classification Pipeline
9. Common NLP Tasks and Approaches
10. Evaluation Metrics
11. Production Considerations

---

## 1. The Evolution of NLP

**The problem**: text is not numbers. Every method in this file is an answer to the question: how do you convert language into a representation a model can compute on? Each era solved a different failure of the previous approach.

| Era | Method | Key Idea | Limitation |
| :--- | :--- | :--- | :--- |
| **Pre-deep learning** | Bag-of-Words, TF-IDF | Word counts + frequency weighting | No word order, no semantics |
| **Static embeddings** | Word2Vec, GloVe, FastText | Dense word vectors from co-occurrence | One vector per word regardless of context |
| **Sequence models** | RNN, LSTM, GRU | Process tokens sequentially with hidden state | Slow to train, struggles with long-range dependencies |
| **Transformers** | BERT, GPT, T5, etc. | Attention over all tokens in parallel | $O(N^2)$ attention cost; long context is expensive |
| **Modern LLMs** | GPT-4, LLaMA, Mistral | Scale + RLHF + instruction tuning | Hallucination, compute cost, alignment |

---

## 2. Bag-of-Words and TF-IDF

### Bag-of-Words

**The problem**: a model needs a fixed-size numerical input. A document is a variable-length string. The simplest conversion is a vector of word counts — but it throws away word order entirely. "Dog bites man" and "Man bites dog" produce identical BoW vectors.

**The core insight**: for many classification tasks (spam detection, topic labeling), word order matters less than word presence. Which words appear is often sufficient to separate classes, even if how they are arranged is lost.

**The mechanics**: represent a document as a sparse vector of word counts over the vocabulary $V$:

$$\text{BoW}(d) \in \mathbb{R}^{|V|}$$

**What breaks**: negation collapses. "not good" and "good" differ by one low-frequency word — BoW gives them nearly identical representations. Bigrams partially fix this but exponentially expand the vocabulary.

---

### TF-IDF

**The problem**: BoW counts every word equally. "The", "is", and "a" appear in every document and carry no discriminative signal — yet they dominate the count vectors.

**The core insight**: weight each word by how distinctive it is across the corpus. A word that appears in only 10 of 10,000 documents is far more informative than one appearing in 9,000. Scale term frequency by log-inverse document frequency.

**The mechanics**:

$$\text{TF-IDF}(t, d) = \underbrace{\frac{f_{t,d}}{\sum_{t'} f_{t',d}}}_{\text{TF}} \times \underbrace{\log \frac{N}{|\{d : t \in d\}|}}_{\text{IDF}}$$

- TF: how often the term appears in this document
- IDF: log of inverse document frequency — penalizes common words like "the"

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

Still useful for: baseline text classifiers, keyword extraction, document similarity at scale.

**What breaks**: TF-IDF is still a bag of words — order is lost. Semantically related words ("car" / "automobile") have zero overlap in their TF-IDF representations. And it cannot generalize to documents with entirely new vocabulary.

---

## 3. Word Embeddings

### The problem

One-hot encoding treats every word as equidistant from every other word. "King" and "queen" are as far apart as "king" and "asparagus". A model trained on these representations must re-learn that kings and queens are related from downstream labeled data — which is expensive and data-hungry.

### The core insight

Words that appear in similar contexts carry similar meanings. Train a shallow network on a proxy task (predict context from word, or word from context). The learned weights become dense vectors where geometric distance approximates semantic proximity.

---

### Word2Vec

**The mechanics**: train on one of two tasks:

- **CBOW (Continuous Bag of Words)**: predict center word from context
- **Skip-gram**: predict context words from center word

$$J = -\frac{1}{T} \sum_{t=1}^{T} \sum_{-c \leq j \leq c, j \neq 0} \log P(w_{t+j} \mid w_t)$$

The result: words that appear in similar contexts end up with similar weight vectors.

$$\text{king} - \text{man} + \text{woman} \approx \text{queen}$$

```python
import gensim.downloader as api

model = api.load("word2vec-google-news-300")
similarity = model.similarity('king', 'queen')
analogy = model.most_similar(positive=['king', 'woman'], negative=['man'])
```

**What breaks**: Word2Vec assigns one vector per word. "Bank" in "river bank" and "bank account" get the same vector — the average of their incompatible contexts. This polysemy problem is the direct motivation for contextual embeddings (BERT).

---

### GloVe

**The problem**: Word2Vec learns from local context windows — it never sees the global co-occurrence statistics of the full corpus directly.

**The core insight**: build the global co-occurrence matrix first, then optimize vectors to predict log co-occurrence counts:

$$J = \sum_{i,j} f(X_{ij}) \left( w_i^T \tilde{w}_j + b_i + \tilde{b}_j - \log X_{ij} \right)^2$$

**What breaks**: GloVe requires materializing the full co-occurrence matrix, which is expensive for very large vocabularies. Like Word2Vec, it produces static embeddings — one vector per word regardless of context.

---

### FastText

**The problem**: Word2Vec and GloVe treat words as atomic units. Morphologically related words ("run", "running", "runner") get independent vectors. Out-of-vocabulary words get nothing.

**The core insight**: represent each word as the sum of its character n-gram embeddings. "running" = embeddings of `<ru`, `run`, `unn`, `nni`, `nin`, `ing`, `ng>`. If "running" was unseen but "runner" was trained, shared n-grams carry meaning.

**What breaks**: character n-grams help morphology but not polysemy. "bank" in its two senses still collapses to one vector.

---

## 4. Sequence Models (RNN, LSTM, GRU)

### Vanilla RNN

**The problem**: fully connected networks and BoW vectors treat each document as a fixed-size unordered bag. They have no mechanism for remembering that a word at position 3 is related to a word at position 47. Sequential structure — the subject that agrees with a later verb — is invisible.

**The core insight**: maintain a hidden state that summarizes everything seen so far. Apply the same function at every step, feeding the previous hidden state forward:

$$h_t = \tanh(W_h h_{t-1} + W_x x_t + b)$$

**What breaks**: the gradient flows back through $W_h$ at every timestep. If singular values of $W_h$ are less than 1, gradients vanish exponentially over long sequences. If greater than 1, they explode. In practice, vanilla RNNs cannot retain information beyond ~10–20 timesteps.

---

### LSTM (Long Short-Term Memory)

**The problem**: RNNs cannot decide what to keep in memory and what to discard. The hidden state is overwritten at every step — facts from 100 timesteps ago are lost.

**The core insight**: introduce a separate cell state $c_t$ with *additive* updates. Learned gates act as soft switches to write to, read from, and erase the cell state. The additive path allows gradients to flow through time without the repeated multiplicative decay of vanilla RNNs.

**The mechanics**:

| Gate | Formula | Role |
| :--- | :--- | :--- |
| **Forget** | $f_t = \sigma(W_f [h_{t-1}, x_t] + b_f)$ | How much of $c_{t-1}$ to keep |
| **Input** | $i_t = \sigma(W_i [h_{t-1}, x_t] + b_i)$ | How much of new candidate to write |
| **Output** | $o_t = \sigma(W_o [h_{t-1}, x_t] + b_o)$ | How much of $c_t$ to expose |

$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$
$$h_t = o_t \odot \tanh(c_t)$$

The additive cell update $c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$ is the key: the gradient path through $c_t$ does not multiply through the same $W_h$ matrix at every step.

**What breaks**: LSTMs are sequential — step $t$ must wait for step $t-1$. This prevents parallelization across the time dimension. Training on long sequences is slow. The hidden state is still a fixed-size bottleneck for very long documents.

---

### GRU (Gated Recurrent Unit)

**The problem**: LSTM has three gates and two state vectors — more parameters than necessary for many tasks.

**The core insight**: merge cell and hidden state into one. Use two gates (reset and update) instead of three:

$$z_t = \sigma(W_z [h_{t-1}, x_t])$$
$$r_t = \sigma(W_r [h_{t-1}, x_t])$$
$$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$

GRU matches LSTM performance on many tasks with fewer parameters.

---

### Why Sequence Models Lost to Transformers

| Problem | Impact |
| :--- | :--- |
| Sequential processing | Cannot be parallelized across time steps — slow on GPUs |
| Long-range dependencies | Even LSTM degrades on very long sequences |
| Fixed-size bottleneck | The hidden state is a single vector summarizing all history |

---

## 5. Encoder-Decoder and Attention

### Encoder-Decoder

**The problem**: tasks like machine translation require variable-length output from variable-length input. A single classifier with fixed output size cannot do this.

**The core insight**: split into two stages. The encoder compresses the input sequence into a context representation. The decoder generates the output sequence, conditioned on that representation.

**What breaks**: the single context vector $h_T$ is a fixed-size bottleneck. For long input sequences, the encoder must compress everything into one vector. Translation quality degrades sharply for sentences longer than ~20 words.

---

### Bahdanau Attention (2015)

**The problem**: the bottleneck is the single context vector. When generating the 40th output word, the decoder needs to attend to the relevant part of the input — not the compressed summary of everything.

**The core insight**: at each decoding step, let the decoder compute a weighted sum over all encoder hidden states. The weights are learned alignment scores — the model learns which input positions matter for each output position.

$$e_{ij} = a(s_{i-1}, h_j) \quad \text{(alignment score)}$$
$$\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_k \exp(e_{ik})} \quad \text{(attention weight)}$$
$$c_i = \sum_j \alpha_{ij} h_j \quad \text{(context vector)}$$

This was the direct predecessor to self-attention in Transformers. The key difference: Bahdanau attention is cross-attention between encoder and decoder states; Transformer self-attention attends within the same sequence.

**What breaks**: attention is computed recurrently — one step at a time. The Transformer computes all attention in parallel, removing the sequential bottleneck.

---

## 6. Transformers in NLP

Full architecture covered in [transformers.md](../components/transformers.md) and [attention.md](../components/attention.md). The NLP-specific points:

---

### BERT

**The problem**: GPT-style left-to-right language modeling only conditions on past tokens. For tasks like question answering and NER, the model needs both left and right context simultaneously to understand a token's meaning.

**The core insight**: mask tokens at random and train the model to predict them using bidirectional context. Every token attends to every other token. The result: rich contextual representations where the same word in different sentences produces different embeddings.

Pre-trained with two tasks:
1. **Masked Language Modeling (MLM)**: predict 15% of randomly masked tokens
2. **Next Sentence Prediction (NSP)**: classify if two sentences are consecutive (later shown less important)

BERT is bidirectional — every token attends to all other tokens. This makes it excellent for classification, NER, QA, and sentence embeddings.

**Fine-tuning**: add a task-specific head on top of the `[CLS]` token, fine-tune on labeled data.

```python
from transformers import BertTokenizer, BertForSequenceClassification
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

inputs = tokenizer("I loved this movie", return_tensors='pt', padding=True, truncation=True)
outputs = model(**inputs)
logits = outputs.logits
```

**What breaks**: BERT's max sequence length is 512 tokens — it cannot process long documents natively. Its CLS token was trained for NSP, making it a weak sentence encoder for similarity tasks (SBERT fixes this). Bidirectionality prevents it from being used directly for text generation.

---

### GPT

**The problem**: BERT is an encoder — it builds representations but cannot generate text. Many tasks (summarization, translation, code generation) require generating new sequences.

**The core insight**: train a decoder-only model on causal language modeling: predict the next token given all previous tokens. Each token only attends to past tokens (masked self-attention). This makes the model naturally generative.

$$L = -\sum_t \log P(w_t \mid w_{<t})$$

GPT-3 and beyond demonstrate in-context learning — the model adapts to new tasks from examples in the prompt without any gradient updates.

**What breaks**: because GPT is causal, it sees only past context. It cannot condition on future tokens, making it weaker than BERT for tasks that require bidirectional understanding (NER, QA over short passages).

---

### T5

**The problem**: different NLP tasks (translation, summarization, classification) require different architectures and fine-tuning procedures. Managing a separate model for each task is expensive.

**The core insight**: every NLP task is text-in, text-out. Frame everything as a text-to-text problem with a prefix that specifies the task. One model, one training objective, all tasks.

- Translation: `translate English to French: The cat sat on the mat`
- Summarization: `summarize: [document]`
- Classification: `sentiment: I loved this film` → `positive`

**What breaks**: the text-to-text framing requires the model to generate valid label strings exactly. Classification tasks need constrained decoding or post-processing to map free-text output to label sets.

---

### Key Architectural Variants

| Model family | Architecture | Primary use |
| :--- | :--- | :--- |
| **BERT, RoBERTa, DeBERTa** | Encoder-only | Classification, NER, QA |
| **GPT, LLaMA, Mistral** | Decoder-only | Generation, instruction following |
| **T5, BART, mT5** | Encoder-decoder | Translation, summarization |

---

## 7. Tokenization

**The problem**: vocabulary-based tokenization (one integer per word) requires a vocabulary of 100,000+ entries to cover English alone. Any unseen word at test time gets no representation. And it cannot represent morphological variants without seeing every form independently.

**The core insight**: split words into reusable subword units. Common substrings like "un-", "-ness", "happ-" appear in thousands of words. Learn embeddings for these units and any word decomposes into known pieces — including words never seen in training.

---

### Byte-Pair Encoding (BPE)

**The mechanics**:
1. Start with individual characters as vocabulary
2. Count frequency of all adjacent pairs
3. Merge the most frequent pair into a new token
4. Repeat until vocabulary reaches target size

"unbelievable" might tokenize as: `un`, `believ`, `able` — reuses common subword units.

Used by: GPT-2, GPT-4, LLaMA, RoBERTa.

**What breaks**: BPE is greedy — merges are based on training corpus frequency. English-dominated corpora produce suboptimal splits for other languages. Numbers and code fragment poorly (each digit often becomes its own token).

---

### WordPiece (BERT)

**The problem**: frequency-based merging produces units that are statistically common but not linguistically meaningful on small corpora.

**The core insight**: merge based on likelihood gain rather than raw frequency — which merge maximizes the probability of the training data under a unigram language model? Subwords after the first start with `##`:

`"unbelievable"` → `un`, `##believ`, `##able`

Used by: BERT, DistilBERT, Electra.

---

### SentencePiece

**The problem**: BPE and WordPiece assume text is pre-tokenized by whitespace. This breaks on Japanese, Chinese, and other languages without whitespace delimiters.

**The core insight**: treat the entire raw byte stream as input. No pre-tokenization. Language-agnostic and supports lossless round-tripping.

Used by: T5, mT5, LLaMA (via BPE on byte sequences), ALBERT.

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokens = tokenizer.encode("unbelievable")
print(tokenizer.convert_ids_to_tokens(tokens))
# ['un', 'believ', 'able']
```

**What breaks**: a large vocabulary means a large embedding table. GPT-4's ~100K-token vocabulary needs ~100K × 768 = 76M parameters for the embedding layer alone. Reducing vocabulary forces longer sequences and more compute.

---

## 8. Text Classification Pipeline

**The problem**: the right representation strategy depends entirely on data volume. With millions of examples, TF-IDF + logistic regression is fast, interpretable, and competitive. With thousands of examples, fine-tuning a pre-trained language model consistently outperforms — but adds GPU cost and latency.

**The core insight**: start with the simplest approach that could work. TF-IDF + logistic regression runs in seconds and provides a strong baseline. Fine-tune a Transformer only when (a) the gap is significant on validation data or (b) semantic understanding is required.

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

**What breaks**: BERT's max input length is 512 tokens. Long documents must be truncated or chunked — and global context is lost. The `[CLS]` token was trained for NSP, making it a weak sentence embedding without further fine-tuning. Learning rate must be small (~2e-5) — standard 1e-3 rates catastrophically forget pre-trained weights.

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

### BLEU

**The problem**: how do you score a translation automatically when many valid translations exist?

**The core insight**: a good translation shares n-grams with the reference. Measure modified n-gram precision with a brevity penalty to prevent trivially short hypotheses.

**What breaks**: BLEU checks surface n-gram overlap. A paraphrase that is semantically perfect but uses different words scores near zero. Unreliable at sentence level; only meaningful at corpus level.

---

### ROUGE

**The core insight**: for summarization, recall matters — did the summary capture the key information? ROUGE measures the fraction of reference n-grams that appear in the hypothesis.

- **ROUGE-N**: n-gram recall between hypothesis and reference
- **ROUGE-L**: longest common subsequence

**What breaks**: ROUGE rewards word-for-word copying. A summary that paraphrases perfectly but uses synonyms scores lower than a verbatim extract.

---

### Perplexity

**The problem**: how do you measure whether a language model is making confident, correct predictions?

**The core insight**: perplexity is the geometric mean of inverse probabilities assigned to each token. A model with perplexity 50 is as uncertain as uniform choice over 50 options at each token.

$$\text{PPL} = \exp\left(-\frac{1}{N} \sum_i \log P(w_i \mid w_{<i})\right)$$

Lower is better.

**What breaks**: perplexity is only comparable across models with the same tokenizer and vocabulary. A model with a larger vocabulary or different tokenization may report lower perplexity without being a better language model.

---

### BERTScore

**The problem**: BLEU and ROUGE treat "automobile" and "car" as completely different. The metric is zero even when the semantic content is identical.

**The core insight**: compute token-level cosine similarity using contextual BERT embeddings. Semantic equivalents receive high similarity even with different surface forms.

**What breaks**: requires a BERT forward pass per hypothesis-reference pair — 100–1000× slower than BLEU/ROUGE. Does not detect factual errors; a plausible but wrong sentence can score highly.

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

**What breaks in production**:

- **Tokenizer mismatch**: inference tokenizer different from training tokenizer — silent degradation
- **Label distribution shift**: model trained on balanced data; production is 95% negative — threshold must be recalibrated
- **Max length truncation**: truncating long documents drops the tail end silently — for some tasks (legal, medical) the conclusion is at the end
- **Batching with padding**: padding to the longest sequence in a batch inflates compute; dynamic padding or bucketed batching mitigates this

> [!TIP]
> **Interview structure:** For any NLP question — (1) **representation**: how you convert text to vectors, (2) **model**: which architecture and why, (3) **evaluation**: what metric and why it matters, (4) **production**: what breaks at scale.
