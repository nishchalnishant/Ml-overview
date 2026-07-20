---
module: Domains
topic: Methods
subtopic: Nlp Fundamentals
status: unread
tags: [deeplearning, ml, methods-nlp-fundamentals]
---
# NLP Fundamentals

Core natural language processing: tokenization, word representations, classical sequence models, and the building blocks that underlie modern LLMs.

---

## Text Pipeline

```
Raw Text → Tokenization → Numericalization → Embedding → Model → Output
```

---

## Tokenization

### The problem

A neural network operates on numbers. A sentence like "unhappiness" must become a sequence of integers before any computation can happen. The trivial solution — assign one integer to every word in a dictionary — breaks immediately: a vocabulary of 100,000 words needs a 100,000-dimensional embedding table, and any word not seen during training (OOV) gets no representation at all.

### The core insight

Split words into reusable subword units. Common substrings like "un-", "-ness", "happ-" appear in thousands of words. If you learn embeddings for these units, you can represent any word as a composition of known pieces — including words never seen in training.

---

### Word-Level

Split on whitespace/punctuation. Simple but vocabulary is large and OOV words get nothing.

### Character-Level

Each character is a token. Vocabulary is tiny (26+) and handles anything, but sequences are very long — "unhappiness" becomes 11 tokens instead of 1-3.

### Subword Tokenization

Current standard. Splits words into meaningful subword units. Balances vocabulary size, sequence length, and OOV handling.

---

### Byte-Pair Encoding (BPE)

**The problem**: how do you discover which subword units are worth keeping? Common substrings should be single tokens; rare character combinations should be split up.

**The core insight**: merge the most frequent adjacent pair of tokens at each step. Start with individual characters. If "un" appears 10,000 times together, merge them into one token "un". Repeat until vocabulary reaches target size. Frequency drives what gets merged — common patterns become atoms, rare patterns stay split.

**The mechanics**:
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

**What breaks**: BPE is greedy — it merges based on frequency in the training corpus. If the corpus is English-dominated, other languages get suboptimal splits. Numbers and code often fragment poorly (each digit becomes its own token).

---

### WordPiece

Similar to BPE but merges based on **likelihood maximization** (not frequency). Uses `##` prefix for continuation tokens.

**The core insight**: instead of "merge the most frequent pair," ask "which merge maximizes the probability of the training data under a unigram language model?" This produces linguistically more meaningful units on small corpora.

```python
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokens = tokenizer.tokenize("unhappiness")
# → ['un', '##happi', '##ness']
```

Used by: BERT, DistilBERT, Electra.

---

### SentencePiece / Unigram LM

**The problem**: BPE and WordPiece both assume text is pre-tokenized by whitespace. This breaks on languages without whitespace delimiters (Japanese, Chinese) and loses the information that spaces encode.

**The core insight**: treat the entire raw byte stream as input. No pre-tokenization step. The tokenizer is language-agnostic and handles any script uniformly. SentencePiece also supports lossless round-tripping: the original string can always be recovered from the tokens.

```python
import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.Load("tokenizer.model")
tokens = sp.EncodeAsPieces("Hello, world!")
ids = sp.EncodeAsIds("Hello, world!")
```

Used by: T5, mT5, LLaMA (via BPE on byte sequences), ALBERT.

---

### Vocabulary Size Tradeoffs

| Vocab Size | Seq Length | OOV handling | Memory |
|-----------|-----------|-------------|--------|
| Small (8K) | Long | Good | Low |
| Medium (32K) | Medium | Good | Medium |
| Large (100K+) | Short | Excellent | High |

GPT-4 uses ~100K; LLaMA uses 32K; BERT uses 30K (WordPiece).

**What breaks**: a large vocabulary means a large embedding table. GPT-4's 100K-token vocabulary needs 100K × 768 = 76M parameters just for the embedding layer. Reducing vocabulary forces longer sequences and more compute.

---

## Word Embeddings

### The problem

One-hot encoding represents each word as a sparse vector with one 1 and everything else 0. A vocabulary of 100,000 words produces 100,000-dimensional inputs. Cosine similarity between any two one-hot vectors is zero — the representation encodes no notion that "king" and "queen" are more related to each other than "king" and "asparagus."

### The core insight

Words that appear in similar contexts should have similar representations. Train a neural network on a proxy task (predict context from word, or word from context). The learned hidden-layer weights are word vectors where geometric distance approximates semantic similarity.

---

### Word2Vec (Mikolov et al., 2013)

**The mechanics**: train a shallow neural network on one of two tasks:
- **Skip-gram**: given a center word, predict surrounding context words — `P(context | word)`
- **CBOW**: given surrounding context, predict the center word — `P(word | context)`

The hidden layer weights become the word embeddings. Words seen in similar contexts end up with similar weight vectors.

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

**Training tricks**:
- **Negative sampling**: instead of softmax over the full vocabulary (expensive), train a binary classifier against `K` random "negative" words
- **Subsampling**: downsample frequent words ("the", "is", "a") proportionally — they appear in so many contexts that they add noise

**What breaks**: Word2Vec assigns one vector per word. "Bank" in "river bank" and "bank account" get the same vector — the average of two very different contexts. This polysemy problem is solved by contextual embeddings.

---

### GloVe (Pennington et al., 2014)

**The problem**: Word2Vec learns from local context windows — it processes the text one window at a time and never sees global co-occurrence statistics directly.

**The core insight**: if two words appear together much more than random, their dot product should reflect that. Build a global co-occurrence matrix first, then optimize word vectors to predict log co-occurrence counts:

`w_i · w̃_j + b_i + b̃_j = log X_{ij}`

```python
embeddings = {}
with open('glove.6B.100d.txt') as f:
    for line in f:
        parts = line.split()
        word, vec = parts[0], np.array(parts[1:], dtype=float)
        embeddings[word] = vec
```

**Word2Vec vs GloVe**: Word2Vec uses local context (sliding window); GloVe uses global corpus statistics (full co-occurrence matrix). In practice: similar performance. GloVe slightly better on analogy tasks; Word2Vec faster to train.

**What breaks**: both are static — one vector per word regardless of context.

---

### FastText (Bojanowski et al., 2017)

**The problem**: Word2Vec and GloVe treat each word as an atomic unit. Words with shared morphology ("run", "running", "runner") get independent vectors despite sharing a root. OOV words get no representation at all.

**The core insight**: represent each word as the sum of its character n-gram embeddings. "running" = sum of embeddings for `<ru`, `run`, `unn`, `nni`, `nin`, `ing`, `ng>`. If "running" was not in training but "runner" was, the n-grams they share carry meaningful signal.

```python
from gensim.models import FastText

model = FastText(sentences, vector_size=100, window=5, min_count=1, epochs=10)
# Handles OOV: builds embedding from character n-grams
vec_oov = model.wv['unhappiness']  # works even if not in vocabulary
```

**What breaks**: FastText is still static — the embedding for "bank" in "river bank" is the same as in "bank account." Character n-grams help with morphology but not polysemy.

---

## Contextual Embeddings

### The problem

Static embeddings assign one vector per word regardless of context. "The bank raised interest rates" and "the fisherman sat on the bank" produce identical representations for "bank" despite completely different meanings. Downstream models must resolve this ambiguity from scratch.

### The core insight

Generate embeddings dynamically, conditioned on the full surrounding context. The vector for "bank" in a financial sentence and the vector for "bank" in a geographic sentence should be different. This requires a model that reads the sentence before producing the embedding.

---

### ELMo (Peters et al., 2018)

Train bidirectional LSTM language models (forward LM + backward LM). Word embedding = weighted sum of all layer outputs. Deep layers capture semantic information; lower layers capture syntactic structure.

```python
import tensorflow_hub as hub

elmo = hub.load("https://tfhub.dev/google/elmo/3")
embeddings = elmo(["I love machine learning"], signature="default", as_dict=True)["elmo"]
# Shape: (batch, seq_len, 1024)
```

**What breaks**: ELMo is LSTM-based — it cannot parallelize across the time dimension. Processing 512 tokens requires 512 sequential steps. BERT's Transformer attention is fully parallel.

---

### BERT Embeddings

Use BERT's hidden states as contextual embeddings. Common approaches:
- Last hidden state: one vector per token
- Concatenation/sum of last 4 layers: richer features
- CLS token: sentence-level representation

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

**What breaks**: BERT's CLS token was trained with a Next Sentence Prediction objective that makes it a weak sentence encoder. For sentence similarity, SBERT (Sentence-BERT) fine-tunes BERT specifically on paraphrase detection, producing much better sentence embeddings.

---

## Sequence-to-Sequence Models

### Encoder-Decoder Architecture

**The problem**: tasks like machine translation require mapping a variable-length input sequence to a variable-length output sequence. A single fixed-output classifier cannot do this — the target length is unknown and varies with the input.

**The core insight**: split the problem into two stages. An encoder compresses the input sequence into a context representation. A decoder generates the output sequence, one token at a time, conditioned on that representation.

```
Encoder: h_T = RNN([x_1, ..., x_T])    (last hidden state)
Decoder: y_t = softmax(W h_t)
         h_t = RNN(y_{t-1}, h_{t-1})   with h_0 = h_T
```

**What breaks**: the single context vector `h_T` is a bottleneck. For long input sequences, the encoder must compress all information into a fixed-size vector. Translation quality degrades sharply for sentences longer than ~20 words.

---

### Seq2Seq + Attention (Bahdanau, 2014)

**The problem**: a single context vector cannot hold the content of a 50-word sentence. When translating the 40th output word, the decoder needs to attend specifically to the relevant part of the input — not the entire compressed summary.

**The core insight**: at each decoding step, let the decoder attend over all encoder hidden states and form a weighted sum. The weights are learned alignment scores — the model learns which input positions matter for each output position.

**The mechanics**:
```
e_{t,s} = attention_score(h_{decoder,t-1}, h_{encoder,s})
α_{t,s} = softmax(e_{t,s})
context_t = Σ_s α_{t,s} h_{encoder,s}
y_t = softmax(W [h_{decoder,t}; context_t])
```

This is the direct precursor to the Transformer's self-attention. The key difference: Bahdanau attention is cross-attention between encoder and decoder states; Transformer self-attention attends within the same sequence.

**What breaks**: attention is computed recurrently — one step at a time. The Transformer computes all attention in parallel, removing the sequential bottleneck.

---

### Beam Search Decoding

**The problem**: greedy decoding picks the single most likely token at each step. "I love" followed by "cats" might be worse overall than "I love" followed by "my cat" even if "cats" had higher probability at step 3 — because "cats" leads to awkward continuations.

**The core insight**: maintain `K` candidate sequences simultaneously. At each step, expand every candidate by all possible next tokens and keep only the top `K` by total log-probability. This explores a much larger portion of the output space than greedy decoding.

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

**Length penalty**: beam search accumulates negative log-probabilities — longer sequences have lower scores simply because they have more terms. Apply length penalty `score / len(seq)^α` (α ≈ 0.6-0.9) to avoid preference for short outputs.

**What breaks**: beam search is still an approximation — the globally optimal sequence may not be in any of the `K` beams. Wider beams improve quality but slow inference linearly. For open-ended generation (story writing, dialogue), beam search produces repetitive, overly safe text — sampling methods (top-k, nucleus) are preferred.

---

## Named Entity Recognition (NER)

### The problem

You want to extract all people, organizations, and locations from a document. This is not a classification problem (one label per document) — it is a structured prediction problem where each token gets a label and entities can span multiple tokens.

### The core insight

Frame NER as sequence labeling. Use the **BIO tagging scheme** to encode span boundaries: B marks the start of an entity, I marks a continuation, O marks a non-entity token. This converts the span extraction problem into token-level classification.

**BIO tagging scheme**:
- B: Beginning of entity
- I: Inside entity (continuation)
- O: Outside (not an entity)

```
"Barack Obama was born in Hawaii"
 B-PER  I-PER  O    O    O  B-LOC
```

### Models

**BiLSTM-CRF**: bidirectional LSTM produces features at each token; CRF layer decodes the globally optimal tag sequence via the Viterbi algorithm. Long-standing strong baseline.

**BERT + linear head**: fine-tune BERT with a token classification head. The current standard approach.

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

**What breaks**: BERT's subword tokenization misaligns with word-level NER labels. "Obama" might become ["Obama"] (one token → one label) or ["O", "##bama"] (two tokens → one label). The standard fix: take the label from the first subword of each word, ignore the rest.

**Datasets**: CoNLL-2003 (English NER benchmark), OntoNotes 5.0.

---

## Text Classification

### The problem

Assign a label to a document. Sentiment analysis, spam detection, topic classification. A document can be thousands of words long. How do you compress it to a single prediction?

### The core insight

The right representation strategy depends on the data volume. With millions of examples, simple TF-IDF + logistic regression is fast, interpretable, and competitive. With thousands of examples and a pre-trained language model available, fine-tuning BERT consistently outperforms.

---

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

**What breaks**: TF-IDF ignores word order. "The movie was not good" and "The movie was good" have nearly identical TF-IDF vectors differing only in the presence/absence of "not." Bigrams (ngram_range=(1,2)) partially mitigate this.

---

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

**What breaks**: BERT's max input length is 512 tokens. Documents longer than this must be truncated or chunked. For long documents, chunking with overlapping windows and aggregating (mean/max pooling) CLS vectors is a common workaround, but it loses global context.

---

## Machine Translation Metrics

### The problem

Human evaluation of translation quality is expensive and slow. You need an automatic metric to compare models, tune hyperparameters, and track regression. But translation is not deterministic — many valid translations exist for any source sentence. A metric must correlate with human judgment without requiring a unique reference.

---

### BLEU

**The core insight**: a good translation shares n-grams with the reference. Precision — what fraction of the hypothesis n-grams appear in the reference — is the core signal. Add a brevity penalty to prevent trivially short hypotheses from scoring highly.

```python
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu

references = [[['the', 'cat', 'sat', 'on', 'the', 'mat']]]
hypothesis = [['the', 'cat', 'is', 'on', 'the', 'mat']]
score = corpus_bleu(references, hypothesis)   # 0 to 1
```

**What breaks**: BLEU is precision-based and surface-level. A paraphrase that is perfectly correct but uses different words scores zero. BLEU correlates with human judgment at corpus level but is unreliable at sentence level.

---

### ROUGE

**The core insight**: for summarization, recall matters more than precision — did the summary capture the key information? ROUGE measures the fraction of reference n-grams that appear in the hypothesis.

- **ROUGE-1**: unigram overlap
- **ROUGE-2**: bigram overlap
- **ROUGE-L**: longest common subsequence

**What breaks**: ROUGE rewards word-for-word overlap. A summary that paraphrases the source perfectly but uses synonyms scores lower than one that copies verbatim.

---

### METEOR

Harmonic mean of precision and recall, with synonymy matching via WordNet. Better correlation with human judgments than BLEU.

---

### BERTScore

**The problem**: BLEU and ROUGE check if the same words appear. A translation that uses "automobile" instead of "car" scores zero on BLEU even though the meaning is identical.

**The core insight**: compute cosine similarity between contextual BERT embeddings of hypothesis and reference tokens. Semantic equivalents that are lexically different still receive high scores.

```python
from bert_score import score

P, R, F1 = score(hypotheses, references, lang='en', model_type='roberta-large')
```

**What breaks**: BERTScore requires a forward pass through a large model for every hypothesis-reference pair. This is 100-1000× slower than BLEU/ROUGE. Also, BERTScore is not human-interpretable — you cannot tell from the score which parts of the translation are wrong.

---

## Dialogue Systems

### The problem

A user asks a question. The system needs to maintain conversational state across multiple turns, track what was said earlier, resolve references ("What's its capital?" — referring to a country mentioned 3 turns ago), and produce coherent, on-topic responses.

### The core insight

Modern end-to-end dialogue: fine-tune a language model on dialogue data. The context window holds the conversation history. The model handles state tracking, reference resolution, and response generation implicitly through attention over the full context.

---

### Pipeline Architecture (Rule-Based)

```
Input → NLU (intent + entity detection) → Dialogue State Tracking → Policy → NLG → Response
```

Explicit, debuggable, but brittle — requires manual rules for every intent.

### End-to-End Neural Dialogue

```
System: You are a helpful assistant.
User: What's the capital of France?
Assistant: Paris.
User: What's its population?
Assistant: About 2.1 million in the city proper.
```

**Key components**:
- **Context management**: concatenate conversation history into the prompt
- **Persona consistency**: system prompt shapes behavior
- **Grounding**: RAG to reduce hallucination in task-oriented dialogue

### Task-Oriented Dialogue

Goal: complete a specific task (book a restaurant, check flight status). Components:
- **NLU**: intent classification + slot filling (extract parameters: "date", "location", "cuisine")
- **Dialogue State Tracking (DST)**: maintain what the user has specified so far
- **Policy**: decide next system action given current state
- **NLG**: generate natural language response

**What breaks**: the explicit pipeline fails when user utterances are ambiguous or off-domain. End-to-end models handle this more gracefully but are harder to debug and may hallucinate task-critical information (wrong restaurant, wrong date).

**Datasets**: MultiWOZ (multi-domain task-oriented dialogue), DailyDialog.

---

## Key Points

- BPE merges most frequent byte pairs iteratively; WordPiece merges by likelihood. Both create subword vocabularies. SentencePiece works on raw text with no whitespace assumption.
- Word2Vec: local context windows. GloVe: global co-occurrence statistics. FastText: extends with character n-grams for OOV handling.
- ELMo = contextual embeddings from bidirectional LSTM. BERT = bidirectional Transformer with masked language modeling.
- Seq2Seq + Bahdanau attention: decoder attends over all encoder hidden states at each step — this is the direct precursor to Transformer self-attention.
- Beam search: maintain top-K hypotheses at each step. Better than greedy but still an approximation. Length penalty prevents preference for short sequences.
- BLEU measures precision of n-gram overlap; ROUGE measures recall; BERTScore uses contextual embedding similarity — semantic not surface-level.
- NER: BIO tagging converts span extraction to token classification. BERT + token classification head is the standard approach.
