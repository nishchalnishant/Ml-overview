# NLP Fundamentals

Core natural language processing: tokenization, word representations, classical sequence models, and the building blocks that underlie modern LLMs.

---

## Text Pipeline

```
Raw Text → Tokenization → Numericalization → Embedding → Model → Output
```

---

## Tokenization

Tokenization splits raw text into discrete units (tokens). The choice of tokenizer determines vocabulary size, OOV handling, and model efficiency.

### Word-Level

Split on whitespace/punctuation. Simple but large vocabulary, no handling of rare/unseen words.

### Character-Level

Each character is a token. Tiny vocabulary (26+), handles any word, but very long sequences.

### Subword Tokenization

Current standard: splits words into meaningful subword units. Balances vocabulary size and sequence length.

#### Byte-Pair Encoding (BPE)

1. Initialize vocabulary with individual characters
2. Count frequency of all adjacent pairs
3. Merge the most frequent pair into a new token
4. Repeat until vocabulary reaches target size

```python
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

trainer = trainers.BpeTrainer(vocab_size=32000, min_frequency=2)
tokenizer.train(files=["corpus.txt"], trainer=trainer)
tokens = tokenizer.encode("unhappiness").tokens
# → ['un', 'happ', 'iness'] (if rare word) or ['unhappiness'] (if common)
```

Used by: GPT-2, GPT-4, RoBERTa, LLaMA.

#### WordPiece

Similar to BPE but merges based on likelihood maximization (not frequency). Uses `##` prefix for continuation tokens.

```python
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokens = tokenizer.tokenize("unhappiness")
# → ['un', '##happi', '##ness']
```

Used by: BERT, DistilBERT, Electra.

#### SentencePiece / Unigram LM

Treats tokenization as a language modeling problem. Works directly on raw text (no pre-tokenization), handles multiple languages and scripts uniformly. Reversible.

```python
import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.Load("tokenizer.model")
tokens = sp.EncodeAsPieces("Hello, world!")
ids = sp.EncodeAsIds("Hello, world!")
```

Used by: T5, mT5, LLaMA (via BPE on byte sequences), ALBERT.

### Vocabulary Size Tradeoffs

| Vocab Size | Seq Length | OOV handling | Memory |
|-----------|-----------|-------------|--------|
| Small (8K) | Long | Good | Low |
| Medium (32K) | Medium | Good | Medium |
| Large (100K+) | Short | Excellent | High |

GPT-4 uses ~100K; LLaMA uses 32K; BERT uses 30K (WordPiece).

---

## Word Embeddings

### Word2Vec (Mikolov et al., 2013)

Train a shallow neural network to predict context from a word (Skip-gram) or predict a word from context (CBOW). The hidden layer weights become word embeddings.

**Skip-gram:** `P(context | word)` — predict surrounding words given center word  
**CBOW:** `P(word | context)` — predict center word from surrounding context

```python
from gensim.models import Word2Vec

sentences = [["the", "cat", "sat", "on", "mat"], ...]
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, 
                 sg=1,        # 1=Skip-gram, 0=CBOW
                 negative=5,  # negative sampling
                 epochs=10)

vec = model.wv['king']                           # (100,)
similar = model.wv.most_similar('king', topn=5)  # [('queen', 0.85), ...]
# Vector arithmetic: king - man + woman ≈ queen
result = model.wv.most_similar(positive=['king', 'woman'], negative=['man'])
```

**Training tricks:**
- **Negative sampling:** Instead of softmax over full vocabulary, train binary classifiers against K random "negative" words
- **Subsampling:** Downsample frequent words (the, is, a) proportionally to their frequency

### GloVe (Pennington et al., 2014)

Train on global word co-occurrence matrix. Objective: dot product of word vectors should equal log co-occurrence probability.

`w_i · w̃_j + b_i + b̃_j = log X_{ij}`

```python
# Pre-trained GloVe
import numpy as np

embeddings = {}
with open('glove.6B.100d.txt') as f:
    for line in f:
        parts = line.split()
        word, vec = parts[0], np.array(parts[1:], dtype=float)
        embeddings[word] = vec
```

**Word2Vec vs GloVe:**
- Word2Vec: local context (sliding window)
- GloVe: global corpus statistics (co-occurrence matrix)
- In practice: similar performance; GloVe slightly better on analogy tasks; Word2Vec faster to train

### FastText (Bojanowski et al., 2017)

Extends Word2Vec with subword (character n-gram) representations. Each word = sum of its character n-gram embeddings.

```python
from gensim.models import FastText

model = FastText(sentences, vector_size=100, window=5, min_count=1, epochs=10)
# Handles OOV: builds embedding from character n-grams
vec_oov = model.wv['unhappiness']  # works even if not in vocabulary
```

**Advantage:** Handles morphologically rich languages and OOV words.

---

## Contextual Embeddings

Static embeddings assign one vector per word. Contextual embeddings produce different representations depending on context ("bank" in river vs finance).

### ELMo (Peters et al., 2018)

Train bidirectional LSTM language models (forward and backward). Word embedding = weighted sum of all layer outputs.

```python
import tensorflow_hub as hub

elmo = hub.load("https://tfhub.dev/google/elmo/3")
embeddings = elmo(["I love machine learning"], signature="default", as_dict=True)["elmo"]
# Shape: (batch, seq_len, 1024)
```

### BERT Embeddings

Use BERT's final hidden states as contextual embeddings. Common approaches:
- Last hidden state
- Concatenation of last 4 layers
- Sum of last 4 layers
- CLS token for sentence representation

```python
from transformers import BertTokenizer, BertModel
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

inputs = tokenizer("I love ML", return_tensors='pt')
with torch.no_grad():
    outputs = model(**inputs)

token_embeddings = outputs.last_hidden_state   # (1, seq_len, 768)
sentence_embedding = outputs.pooler_output     # (1, 768) — CLS token
```

---

## Sequence-to-Sequence Models

### Encoder-Decoder Architecture

Encode source sequence to a fixed-length context vector; decode to target sequence.

```
Encoder: h_T = RNN([x_1, ..., x_T])    (last hidden state)
Decoder: y_t = softmax(W h_t)
         h_t = RNN(y_{t-1}, h_{t-1})   with h_0 = h_T
```

**Bottleneck problem:** Single context vector is a bottleneck for long sequences. Attention solves this.

### Seq2Seq + Attention (Bahdanau, 2014)

At each decoder step, compute attention over all encoder hidden states.

```
e_{t,s} = attention_score(h_{decoder,t-1}, h_{encoder,s})
α_{t,s} = softmax(e_{t,s})
context_t = Σ_s α_{t,s} h_{encoder,s}
y_t = softmax(W [h_{decoder,t}; context_t])
```

This is the precursor to the Transformer's self-attention.

### Beam Search Decoding

Greedy decoding picks the single most likely token at each step (suboptimal). Beam search maintains the K most likely partial sequences.

```python
def beam_search(model, start_token, beam_width=4, max_len=50):
    beams = [(0.0, [start_token])]   # (score, sequence)
    for _ in range(max_len):
        candidates = []
        for score, seq in beams:
            if seq[-1] == EOS_TOKEN:
                candidates.append((score, seq))
                continue
            logits = model(seq)
            log_probs = F.log_softmax(logits[-1], dim=-1)
            top_probs, top_ids = log_probs.topk(beam_width)
            for prob, tok in zip(top_probs, top_ids):
                candidates.append((score + prob.item(), seq + [tok.item()]))
        beams = sorted(candidates, key=lambda x: x[0], reverse=True)[:beam_width]
    return max(beams, key=lambda x: x[0])[1]
```

**Length penalty:** Beam search favors shorter sequences (less accumulated negative log-prob). Apply length penalty: `score / len(seq)^α`.

---

## Named Entity Recognition (NER)

Identify and classify named entities (persons, organizations, locations, dates) in text.

**BIO tagging scheme:**
- B: Beginning of entity
- I: Inside entity (continuation)
- O: Outside (not an entity)

```
"Barack Obama was born in Hawaii"
 B-PER  I-PER  O    O    O  B-LOC
```

### Models

**CRF on top of features:** Classical approach — Viterbi decoding for globally optimal tag sequence.

**BiLSTM-CRF:** Bidirectional LSTM features + CRF decoding layer. Long-standing strong baseline.

**BERT + linear head:** Fine-tune BERT with a token classification head.

```python
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")

text = "Barack Obama was born in Hawaii."
inputs = tokenizer(text, return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits

predictions = logits.argmax(dim=-1)
labels = [model.config.id2label[p.item()] for p in predictions[0]]
```

**Datasets:** CoNLL-2003 (English NER benchmark), OntoNotes 5.0.

---

## Text Classification

Assign labels to documents. Spans: binary sentiment → fine-grained multi-class → multi-label.

### Classical Pipeline

```
TF-IDF features → Logistic Regression / SVM / Naive Bayes
```

```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=50000, ngram_range=(1,2), sublinear_tf=True)),
    ('clf', LogisticRegression(C=1.0, max_iter=1000)),
])
pipe.fit(X_train, y_train)
```

### BERT Fine-Tuning

```python
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=5)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    learning_rate=2e-5,
    warmup_ratio=0.1,
)
trainer = Trainer(model=model, args=training_args, train_dataset=train_ds, eval_dataset=eval_ds)
trainer.train()
```

---

## Machine Translation Metrics

**BLEU (Bilingual Evaluation Understudy):** Precision-based n-gram overlap between hypothesis and reference(s). Includes brevity penalty.

```python
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu

references = [[['the', 'cat', 'sat', 'on', 'the', 'mat']]]
hypothesis = [['the', 'cat', 'is', 'on', 'the', 'mat']]
score = corpus_bleu(references, hypothesis)   # 0 to 1
```

**ROUGE (Recall-Oriented Understudy for Gisting Evaluation):** Used for summarization. Measures recall of n-grams.
- ROUGE-1: Unigram overlap
- ROUGE-2: Bigram overlap
- ROUGE-L: Longest Common Subsequence

**METEOR:** Harmonic mean of precision and recall, with synonymy matching. Better correlation with human judgments than BLEU.

**BERTScore:** Compute cosine similarity between contextual BERT embeddings of hypothesis and reference tokens. Semantic, not surface-level.

```python
from bert_score import score

P, R, F1 = score(hypotheses, references, lang='en', model_type='roberta-large')
```

---

## Dialogue Systems

### Pipeline Architecture (Rule-Based)

```
Input → NLU (intent + entity detection) → Dialogue State Tracking → Policy → NLG → Response
```

### End-to-End Neural Dialogue

Modern approach: fine-tune a language model (GPT, LLaMA) on dialogue data.

```
System: You are a helpful assistant.
User: What's the capital of France?
Assistant: Paris.
User: What's its population?
Assistant: About 2.1 million in the city proper.
```

**Key components:**
- **Context management:** Maintain conversation history
- **Persona consistency:** System prompt shapes assistant behavior
- **Grounding:** RAG to prevent hallucination in task-oriented dialogue

### Task-Oriented Dialogue

Goal: complete a specific task (book restaurant, check weather). Uses:
- **NLU:** Intent classification + slot filling
- **Dialogue State Tracking (DST):** Track what the user has specified so far
- **Policy:** Decide next system action
- **NLG:** Generate natural language response

**Datasets:** MultiWOZ (multi-domain task-oriented dialogue), DailyDialog.

---

## Key Interview Points

- BPE merges most frequent byte pairs iteratively; WordPiece merges by likelihood. Both create subword vocabularies.
- Word2Vec: local context; GloVe: global co-occurrence. FastText: extends with character n-grams (handles OOV).
- ELMo = contextual embeddings from BiLSTM; BERT = bidirectional Transformer masked language modeling.
- Seq2Seq + attention: decoder attends over all encoder states at each step — precursor to Transformer.
- Beam search: maintain top-K hypotheses at each decoding step; better than greedy but still approximation.
- BLEU measures precision of n-gram overlap; ROUGE measures recall; BERTScore uses semantic similarity.
- NER: BIO tagging scheme; BERT + token classification head is the standard approach.
