# NLP Advanced

Advanced NLP techniques spanning summarization, semantic similarity, inference, coreference, dependency parsing, and relation extraction — the layer above transformers that powers real-world text understanding systems.

---

## Abstractive Summarization

Abstractive summarization generates new text that captures the meaning of the source, rather than copying spans directly. It requires the model to compress, paraphrase, and sometimes infer.

### Seq2Seq Models

All major abstractive summarizers share the encoder-decoder (Seq2Seq) backbone:

```
Source Document → [Encoder] → Context Representations → [Decoder] → Summary
```

| Model | Architecture | Pre-training objective | Strength |
|-------|-------------|----------------------|----------|
| BART | Transformer encoder-decoder | Denoising (reconstruct corrupted text) | Strong on news summarization |
| T5 | Transformer encoder-decoder | Text-to-text on C4 | General-purpose, instruction-tunable |
| Pegasus | Transformer encoder-decoder | Gap-sentence generation (GSG) | Purpose-built for summarization |

### BART Pre-training (Denoising)

BART (Lewis et al., 2019) corrupts input text with a mix of noise functions and trains the decoder to reconstruct the original:

- **Token masking** — replace tokens with `[MASK]`
- **Token deletion** — remove random tokens; model must infer positions
- **Text infilling** — replace arbitrary spans with single `[MASK]` token
- **Sentence permutation** — shuffle sentence order
- **Document rotation** — rotate to start from a random token

The denoising objective forces the encoder to build rich representations and the decoder to learn fluent text generation — both critical for summarization.

### HuggingFace Pipeline

```python
from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

document = """
NASA's James Webb Space Telescope has captured the deepest infrared image of the
universe ever taken. The image shows thousands of galaxies, including some of the
faintest objects ever observed. The telescope, launched in December 2021, is designed
to observe the universe in infrared light, allowing scientists to see through dust
clouds and observe the formation of the first stars and galaxies.
"""

result = summarizer(document, max_length=80, min_length=20, do_sample=False)
print(result[0]["summary_text"])
# → "NASA's James Webb Space Telescope has captured the deepest infrared image of
#    the universe. The image shows thousands of galaxies, some the faintest ever observed."
```

### Fine-tuning Sketch

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer
from transformers import Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from datasets import load_dataset

model_name = "facebook/bart-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

dataset = load_dataset("cnn_dailymail", "3.0.0")

def preprocess(batch):
    inputs = tokenizer(
        batch["article"],
        max_length=1024,
        truncation=True,
        padding="max_length"
    )
    with tokenizer.as_target_tokenizer():
        targets = tokenizer(
            batch["highlights"],
            max_length=128,
            truncation=True,
            padding="max_length"
        )
    inputs["labels"] = targets["input_ids"]
    return inputs

tokenized = dataset.map(preprocess, batched=True, remove_columns=dataset["train"].column_names)

args = Seq2SeqTrainingArguments(
    output_dir="bart-cnn-finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=500,
    predict_with_generate=True,    # use beam search during eval
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    fp16=True,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
)

trainer.train()
```

**Key generation parameters:**
- `num_beams=4` — beam search width
- `length_penalty=2.0` — penalize short summaries (>1 favors longer)
- `no_repeat_ngram_size=3` — prevent repetitive output

---

## Extractive Summarization

Extractive summarization scores and selects the most salient sentences directly from the source, preserving original wording. No text is generated — only sentences are ranked and chosen.

### TextRank

TextRank (Mihalcea & Tarau, 2004) applies PageRank to a sentence similarity graph:

1. Represent each sentence as a TF-IDF or embedding vector
2. Build a fully connected graph: nodes = sentences, edge weight = cosine similarity
3. Run PageRank until convergence; high-PageRank sentences are central/salient
4. Return top-K sentences in original document order

```python
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def textrank_summarize(text: str, top_n: int = 3) -> str:
    import nltk
    sentences = nltk.sent_tokenize(text)

    # Build TF-IDF sentence vectors
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform(sentences)
    sim_matrix = cosine_similarity(tfidf)

    # PageRank iteration
    n = len(sentences)
    scores = np.ones(n) / n
    damping = 0.85
    for _ in range(100):
        new_scores = (1 - damping) / n + damping * (sim_matrix @ scores)
        if np.allclose(scores, new_scores, atol=1e-6):
            break
        scores = new_scores

    # Select top sentences in original order
    ranked = np.argsort(scores)[::-1][:top_n]
    selected = sorted(ranked)
    return " ".join(sentences[i] for i in selected)
```

### BERTSum

BERTSum (Liu & Lapata, 2019) wraps BERT with a sentence-level binary classifier:

```
[CLS] sent_1 [SEP] [CLS] sent_2 [SEP] ... [CLS] sent_n [SEP]
```

Each `[CLS]` token representation is fed to a sigmoid classifier predicting whether that sentence belongs in the summary. The model is fine-tuned end-to-end.

```python
# Using PreSumm / BERTSum via HuggingFace community checkpoint
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Simplified sentence scoring with BERT [CLS] representations
from transformers import BertModel, BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert = BertModel.from_pretrained("bert-base-uncased")

def score_sentences(sentences: list[str]) -> list[float]:
    scores = []
    for sent in sentences:
        inputs = tokenizer(sent, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            out = bert(**inputs)
        cls_vec = out.last_hidden_state[:, 0, :]  # [CLS] token
        # In full BERTSum, this feeds a learned linear classifier
        # Here: use norm as a proxy score for demonstration
        scores.append(cls_vec.norm().item())
    return scores
```

**Extractive vs Abstractive trade-off:**

| | Extractive | Abstractive |
|--|------------|-------------|
| Faithfulness | High (verbatim) | Can hallucinate |
| Fluency | Choppy at boundaries | Natural, coherent |
| Compression | Limited by sentence granularity | Flexible |
| Speed | Fast | Slower (generation) |

---

## Summarization Metrics

### ROUGE (Lin, 2004)

ROUGE measures recall of reference n-grams in the generated summary.

```
ROUGE-N Recall    = |overlap of N-grams| / |N-grams in reference|
ROUGE-N Precision = |overlap of N-grams| / |N-grams in system output|
ROUGE-N F1        = harmonic mean of precision and recall
```

- **ROUGE-1** — unigram overlap (content coverage)
- **ROUGE-2** — bigram overlap (fluency, phrase preservation)
- **ROUGE-L** — longest common subsequence (order-sensitive)

```python
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(
    ["rouge1", "rouge2", "rougeL"],
    use_stemmer=True
)

reference = "The cat sat on the mat near the window."
hypothesis = "A cat was sitting on the mat by the window."

scores = scorer.score(reference, hypothesis)
for key, val in scores.items():
    print(f"{key}: P={val.precision:.3f}  R={val.recall:.3f}  F={val.fmeasure:.3f}")

# rouge1: P=0.750  R=0.750  F=0.750
# rouge2: P=0.500  R=0.500  F=0.500
# rougeL: P=0.625  R=0.625  F=0.625
```

**Batch evaluation:**

```python
import numpy as np
from rouge_score import rouge_scorer

def evaluate_summaries(references: list[str], hypotheses: list[str]) -> dict:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    results = {"rouge1": [], "rouge2": [], "rougeL": []}

    for ref, hyp in zip(references, hypotheses):
        s = scorer.score(ref, hyp)
        for key in results:
            results[key].append(s[key].fmeasure)

    return {k: np.mean(v) for k, v in results.items()}
```

### BERTScore

Computes token-level cosine similarity between reference and hypothesis using contextual BERT embeddings. More robust to paraphrasing than ROUGE.

```python
from bert_score import score as bert_score

references = ["The cat sat on the mat."]
hypotheses = ["A cat was resting on the rug."]

P, R, F1 = bert_score(hypotheses, references, lang="en", verbose=False)
print(f"BERTScore F1: {F1.mean().item():.4f}")
```

### Why ROUGE Alone Is Insufficient

| Issue | Example |
|-------|---------|
| Penalizes valid paraphrases | "automobile" vs "car" — ROUGE=0, but semantically correct |
| Rewards extractive copying | Copy source sentences verbatim → high ROUGE, no compression |
| Ignores factual consistency | "Biden won in 2016" → high ROUGE if reference contains "won in 2016" |
| No fluency signal | Shuffled n-grams can score well |

**Factual consistency metrics:**

- **QAEval** — generates QA pairs from reference, checks if system summary answers them correctly
- **FactCC** — entailment-based: classify each generated sentence as supported/contradicted/neutral with respect to source
- **SummaC** — segment-level NLI: compute entailment probability between each source segment and each summary sentence; aggregate

These metrics measure whether the summary is *faithful to the source*, independent of surface overlap with the reference.

---

## Semantic Similarity and Sentence Embeddings

### SBERT (Sentence-BERT)

Standard BERT produces contextual token embeddings but is expensive for pairwise similarity: comparing N sentences naively requires O(N²) forward passes through a cross-encoder.

SBERT (Reimers & Gurevych, 2019) trains a siamese/triplet BERT network to produce fixed-size sentence embeddings directly comparable via cosine similarity:

```
Sentence A → BERT → Mean Pool → u (768-dim)
Sentence B → BERT → Mean Pool → v (768-dim)

Training objective: cosine similarity loss on sentence pairs
  L = 1 - cos(u, v)    for similar pairs
  L = max(0, cos(u,v)) for dissimilar pairs
```

**Mean pooling** over token embeddings (not just `[CLS]`) proved more informative experimentally.

At inference: encode corpus once → store embeddings → query with a single vector → dot-product search. O(N) instead of O(N²).

### Cosine Similarity Search

```python
from sentence_transformers import SentenceTransformer, util
import torch

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Corpus
corpus = [
    "A man is eating a piece of bread.",
    "The girl is carrying a baby.",
    "A man is riding a horse.",
    "A woman is playing violin.",
    "Two men pushed carts through the woods.",
    "A man is riding a white horse on an enclosed piece of land.",
    "A monkey is playing drums.",
    "A cheetah is running behind its prey.",
]

query = "A person on a horse jumps over a broken-down airplane."

# Encode
corpus_embeddings = model.encode(corpus, convert_to_tensor=True)   # (8, 384)
query_embedding = model.encode(query, convert_to_tensor=True)       # (384,)

# Cosine similarity search
hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=3)

print(f"Query: {query}\n")
for hit in hits[0]:
    print(f"  [{hit['score']:.4f}] {corpus[hit['corpus_id']]}")

# [0.6221] A man is riding a white horse on an enclosed piece of land.
# [0.5935] A man is riding a horse.
# [0.1869] Two men pushed carts through the woods.
```

**Bi-encoder vs Cross-encoder:**

| | Bi-encoder (SBERT) | Cross-encoder |
|--|--------------------|-|
| Speed | Fast — pre-compute embeddings | Slow — full attention over pair |
| Accuracy | Slightly lower | Higher (full interaction) |
| Use case | Retrieval, ANN search | Re-ranking top-K results |
| Scalability | O(N) | O(N) pairs at query time |

---

## Paraphrase Detection

Paraphrase detection determines whether two sentences express the same meaning. It is a binary classification over sentence pairs.

### Datasets

- **PAWS** (Zhang et al., 2019) — hard paraphrase pairs constructed by word swapping + back-translation; adversarial for surface-overlap models
- **QQP** (Quora Question Pairs) — 400K question pairs, binary label
- **MRPC** — Microsoft Research Paraphrase Corpus

### Fine-tuned NLI Models

NLI models trained on SNLI/MultiNLI can be repurposed: if a model predicts *entailment* in both directions (A→B and B→A), the pair is paraphrastic.

More directly, fine-tune a cross-encoder on PAWS/QQP:

```python
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.cross_encoder import CrossEncoder
from torch.utils.data import DataLoader

# Cross-encoder approach — highest accuracy
cross_encoder = CrossEncoder("cross-encoder/stsb-distilroberta-base")

pairs = [
    ("How are you?", "What is your state of being?"),
    ("How are you?", "What is the capital of France?"),
]

scores = cross_encoder.predict(pairs)
print(scores)
# [0.87, 0.03]  — high score means paraphrase
```

**Bi-encoder for approximate paraphrase at scale:**

```python
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2")

sent_a = "The cat sat on the mat."
sent_b = "A feline rested on the rug."
sent_c = "The economy grew by 3% last quarter."

embs = model.encode([sent_a, sent_b, sent_c])
sim_ab = util.cos_sim(embs[0], embs[1]).item()
sim_ac = util.cos_sim(embs[0], embs[2]).item()

print(f"A–B similarity: {sim_ab:.3f}")   # 0.841
print(f"A–C similarity: {sim_ac:.3f}")   # 0.012
```

**Trade-offs recap:**
- Use bi-encoder when filtering millions of pairs (recall phase)
- Use cross-encoder to re-rank top candidates (precision phase)
- The two-stage pipeline gives near cross-encoder accuracy at bi-encoder scale

---

## Natural Language Inference (NLI)

NLI (also called Recognizing Textual Entailment / RTE) classifies the relationship between a *premise* and a *hypothesis*:

| Label | Meaning | Example |
|-------|---------|---------|
| **Entailment** | Premise logically implies hypothesis | P: "A dog is running." H: "An animal is moving." |
| **Contradiction** | Premise logically contradicts hypothesis | P: "A dog is running." H: "No animal is present." |
| **Neutral** | Neither — cannot be inferred either way | P: "A dog is running." H: "The dog is brown." |

### Datasets

- **SNLI** (Bowman et al., 2015) — 570K crowd-sourced pairs from image captions. Clean but narrow domain
- **MultiNLI** (Williams et al., 2018) — 433K pairs across 10 genres (fiction, government, telephone, travel…). More generalizable
- **ANLI** (Nie et al., 2020) — adversarially collected, much harder; three rounds of increasing difficulty

### Model Architecture

Fine-tune a transformer encoder on the [CLS] representation of concatenated premise and hypothesis:

```
[CLS] premise [SEP] hypothesis [SEP] → linear(768, 3) → softmax
```

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")
model = AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli")
# Labels: 0=contradiction, 1=neutral, 2=entailment

premise = "A soccer game with multiple males playing."
hypothesis = "Some men are playing a sport."

inputs = tokenizer(premise, hypothesis, return_tensors="pt", truncation=True)
with torch.no_grad():
    logits = model(**inputs).logits

probs = torch.softmax(logits, dim=-1)
label_map = {0: "contradiction", 1: "neutral", 2: "entailment"}
pred = label_map[probs.argmax().item()]
print(f"Prediction: {pred} (probs: {probs.squeeze().tolist()})")
# Prediction: entailment (probs: [0.003, 0.021, 0.976])
```

### Applications

- **Fact-checking** — premise = article claim, hypothesis = fact to verify
- **QA** — verify if a passage entails a candidate answer
- **Zero-shot classification** — use hypothesis templates (see next section)
- **Summarization faithfulness** — check if summary sentences are entailed by the source

---

## Zero-Shot Classification with NLI

The core insight: phrase each candidate label as a hypothesis — "This example is about {label}" — and use the entailment probability as a classification score. No labeled examples required.

```
Premise  = text to classify
Hypothesis = "This text is about politics."
Entailment probability → score for label "politics"
```

Run for all labels, softmax over entailment scores → zero-shot classifier.

### HuggingFace Pipeline

```python
from transformers import pipeline

classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)

text = "The Federal Reserve raised interest rates for the third consecutive time, citing persistent inflation pressures across the economy."

labels = ["economics", "sports", "technology", "politics", "science"]

result = classifier(text, candidate_labels=labels)
print("Label scores:")
for label, score in zip(result["labels"], result["scores"]):
    print(f"  {label:15s}: {score:.4f}")

# economics      : 0.8231
# politics       : 0.1124
# technology     : 0.0312
# science        : 0.0198
# sports         : 0.0135
```

**Multi-label mode** (each label independent, no softmax across labels):

```python
result = classifier(
    text,
    candidate_labels=labels,
    multi_label=True   # each label scored independently
)
```

### Custom Hypothesis Templates

```python
# Default: "This example is {label}."
# Custom template for sentiment:
result = classifier(
    "The movie was an absolute masterpiece.",
    candidate_labels=["positive", "negative", "neutral"],
    hypothesis_template="The sentiment of this review is {}."
)
```

**Limitations:**
- Hypothesis quality matters enormously — poorly framed hypotheses hurt badly
- Slower than trained classifiers (one NLI forward pass per label)
- NLI models trained on balanced SNLI/MultiNLI may not generalize to all domains

---

## Coreference Resolution

Coreference resolution identifies when different expressions (mentions) in a text refer to the same real-world entity, and clusters them together.

```
"Maria bought a house. She loves it. The property is near downtown."
  └── {Maria, She}          {a house, it, The property}
```

### Pipeline

1. **Mention detection** — find all candidate noun phrases, pronouns, and named entities
2. **Mention pair scoring** — score likelihood that two mentions are coreferent
3. **Clustering** — group coreferent mentions (entity clusters)

### Models

**Stanford CoreNLP** — rule-based + statistical sieve pipeline. Multi-pass: deterministic rules first (string match, syntactic rules), then statistical models.

**spaCy + neuralcoref** — adds neural coreference resolution on top of spaCy's NLP pipeline. Simple API but no longer actively maintained.

**SpanBERT for Coreference (Lee et al., 2018; updated with SpanBERT 2020)** — current gold standard. Encodes all spans in a document, uses a span-pair classifier trained end-to-end:

```
SpanBERT encoder → span representations → antecedent scoring → clusters
```

```python
import spacy

# Using spacy-experimental coref (successor to neuralcoref)
nlp = spacy.load("en_coreference_web_trf")

doc = nlp("Maria bought a house. She loves it. The property is near downtown.")

for cluster_id, cluster in doc.spans.items():
    if cluster_id.startswith("coref_clusters_"):
        print(f"Cluster: {[span.text for span in cluster]}")

# Cluster: ['Maria', 'She']
# Cluster: ['a house', 'it', 'The property']
```

**Applications:**
- Information extraction (resolving pronouns to named entities before extraction)
- Reading comprehension (understanding "he" and "she" in QA context)
- Summarization (avoid pronoun dangling in output)

---

## Dependency Parsing

Dependency parsing assigns a syntactic structure to a sentence by identifying directed *head-dependent* relations between words. Every word (except the root) has exactly one head.

```
nsubj     dobj
 ↑         ↑
"The cat ate the fish quickly"
                              ↑
                            advmod
```

### Universal Dependencies (UD)

UD defines a cross-lingual standard set of dependency labels. Key relations:

| Relation | Meaning | Example |
|----------|---------|---------|
| `nsubj` | Nominal subject | *cat* in "The **cat** eats" |
| `dobj` | Direct object | *fish* in "eats the **fish**" |
| `amod` | Adjectival modifier | *big* in "the **big** cat" |
| `advmod` | Adverbial modifier | *quickly* in "eats **quickly**" |
| `prep` / `case` | Prepositional attachment | *in* in "sits **in** the box" |
| `conj` | Conjunct | second verb in coordination |

### spaCy Dependency Parse

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The quick brown fox jumps over the lazy dog.")

for token in doc:
    print(f"{token.text:12s}  head={token.head.text:12s}  dep={token.dep_:10s}  pos={token.pos_}")

# The          head=fox           dep=det        pos=DET
# quick        head=fox           dep=amod       pos=ADJ
# brown        head=fox           dep=amod       pos=ADJ
# fox          head=jumps         dep=nsubj      pos=NOUN
# jumps        head=jumps         dep=ROOT       pos=VERB
# over         head=jumps         dep=prep       pos=ADP
# the          head=dog           dep=det        pos=DET
# lazy         head=dog           dep=amod       pos=ADJ
# dog          head=over          dep=pobj       pos=NOUN
```

**Visualize:**

```python
from spacy import displacy
displacy.serve(doc, style="dep")  # opens browser at localhost:5000
```

### Applications

- **Relation extraction** — "Apple acquired Beats" → `nsubj(acquired, Apple)` + `dobj(acquired, Beats)`
- **Semantic Role Labeling (SRL)** — identify who did what to whom
- **Question answering** — parse question structure to match against parsed passage
- **Grammar correction** — detect malformed dependency structures

---

## Relation Extraction

Relation extraction (RE) identifies *semantic relationships* between named entities in text.

```
"Elon Musk founded SpaceX in 2002."
 └─ (Elon Musk) ──[founded]-→ (SpaceX)
```

### Task Formulation

Given: a sentence + two marked entity spans (subject, object)
Output: relation type from a predefined schema, or NONE

### Supervised Approach — TACRED

TACRED (Zhang et al., 2017) is the standard benchmark: 42 relation types + no_relation, ~100K examples from news/web.

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Fine-tuned RE model (TACRED)
tokenizer = AutoTokenizer.from_pretrained("DFKI-SLT/mrebel-large")  # example

# Entity marking: wrap entities with special tokens
text = "[SUBJ-PERSON] Elon Musk [/SUBJ-PERSON] founded [OBJ-ORG] SpaceX [/OBJ-ORG] in 2002."

# The model learns that [SUBJ-*] and [OBJ-*] mark the relation arguments
inputs = tokenizer(text, return_tensors="pt")
# → classification over 42+1 relation types
```

**Standard RE Pipeline:**

```python
import spacy
from transformers import pipeline

nlp = spacy.load("en_core_web_sm")

# Step 1: NER to find entities
doc = nlp("Jeff Bezos founded Amazon in Bellevue, Washington.")
entities = [(ent.text, ent.label_, ent.start_char, ent.end_char) for ent in doc.ents]
print(entities)
# [('Jeff Bezos', 'PERSON', ...), ('Amazon', 'ORG', ...), ('Bellevue', 'GPE', ...), ('Washington', 'GPE', ...)]

# Step 2: For each entity pair → RE model
# (entity pair enumeration + RE model call omitted for brevity)
```

### Distant Supervision

When labeled data is scarce, distant supervision automatically generates training data:

1. Take a knowledge base (Freebase, Wikidata) with known relation triples
2. Find all sentences in a text corpus that mention both entities
3. Assume those sentences express the relation (noisy but scalable)

**Noise problem:** "Obama was born in Honolulu" and "Obama visited Honolulu" both match `(Obama, bornIn, Honolulu)` but only the first expresses the relation. Multi-instance learning (aggregate over all sentences for an entity pair) mitigates this.

### LLM-Based Extraction

Modern approach — prompt an LLM to extract relations directly:

```python
from openai import OpenAI

client = OpenAI()

text = "Marie Curie was born in Warsaw and later moved to Paris, where she conducted her Nobel Prize-winning research at the University of Paris."

prompt = f"""Extract all (subject, relation, object) triples from the text below.
Return JSON list of objects with keys: subject, relation, object.

Text: {text}"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    response_format={"type": "json_object"},
)

import json
triples = json.loads(response.choices[0].message.content)
# [
#   {"subject": "Marie Curie", "relation": "born_in",      "object": "Warsaw"},
#   {"subject": "Marie Curie", "relation": "worked_at",    "object": "University of Paris"},
#   {"subject": "Marie Curie", "relation": "won",          "object": "Nobel Prize"},
# ]
```

**Supervised vs LLM extraction:**

| | Supervised (TACRED) | Distant Supervision | LLM-based |
|--|---------------------|---------------------|-----------|
| Accuracy | High (in-domain) | Noisy | High (flexible) |
| Schema | Fixed | Fixed | Open / flexible |
| Data need | Labeled pairs | KB + corpus | None (zero-shot) |
| Speed | Fast inference | Fast inference | Slow + costly |
| Generalization | Poor (new relations) | Poor | Good |

---

## Key Interview Points

**Abstractive vs Extractive Summarization**
- Extractive: score and select sentences; fast, faithful, but grammatically choppy at boundaries
- Abstractive: generate new text; more fluent and flexible, but can hallucinate
- BART pre-training: denoising autoencoder — corrupts text (masking, deletion, permutation) and trains decoder to reconstruct; gives strong generation prior

**ROUGE Limitations**
- ROUGE rewards n-gram overlap with reference — penalizes valid paraphrases, rewards copying
- High ROUGE does not imply factual correctness or fluency
- Complement with BERTScore (semantic), FactCC/SummaC (faithfulness), human evaluation

**SBERT vs Plain BERT for Similarity**
- Plain BERT: O(N²) cross-encoder passes for N-sentence comparison
- SBERT: encode once → cached embeddings → dot-product search in O(N); 5× faster at minor accuracy cost
- Mean pooling over token embeddings outperforms [CLS]-only

**NLI as Universal Tool**
- NLI models (fine-tuned on MultiNLI) power zero-shot classification via hypothesis templates
- Any classification problem can be framed as entailment: "This text is about {label}."
- Also used for fact-checking, summarization faithfulness (FactCC, SummaC), and paraphrase detection

**Zero-Shot Classification**
- Use entailment probability from an NLI model as label score — no training data needed
- Quality of hypothesis template is the main lever; "This text is about X" works broadly
- Multi-label mode scores each label independently (no softmax competition)

**Coreference Resolution**
- Three stages: mention detection, pairwise scoring, clustering
- SpanBERT end-to-end model is the current standard
- Critical for downstream tasks: if a sentence contains "he acquired it," RE cannot fire without resolving pronouns first

**Dependency Parsing**
- Every word has one head (except root); labeled directed edges (nsubj, dobj, amod…)
- Universal Dependencies enables cross-lingual consistency
- Used upstream of relation extraction, SRL, grammar correction

**Relation Extraction**
- Supervised (TACRED): fixed schema, high precision, doesn't generalize to new relations
- Distant supervision: scalable but noisy; multi-instance learning helps
- LLM-based: zero-shot, open-schema, flexible — best for prototyping or low-resource scenarios
- Real pipelines: NER → entity pair enumeration → RE model → knowledge graph population

**Cross-encoder vs Bi-encoder**
- Bi-encoder (SBERT): encode independently → cosine similarity; scales to millions, slightly lower accuracy
- Cross-encoder: full attention over concatenated pair; highest accuracy, O(N) at query time
- Production pattern: bi-encoder for recall (retrieve top-K), cross-encoder for re-ranking (precision)
