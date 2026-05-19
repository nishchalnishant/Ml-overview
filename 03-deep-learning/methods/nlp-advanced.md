# NLP Advanced

Advanced NLP: summarization, semantic similarity, inference, coreference, dependency parsing, and relation extraction — the layer above Transformers that powers real-world text understanding systems.

---

## Abstractive Summarization

### The problem

A news article is 800 words. You need a 3-sentence summary. Extractive methods can select the 3 most salient sentences verbatim — but those sentences rarely form a coherent summary when stitched together. They repeat context, reference entities never introduced, and often contain the least informative sentence in a paragraph that happens to contain a few high-TF-IDF terms.

### The core insight

Summarization is compression + paraphrase. A model must learn to read the whole document, identify what matters, and generate new fluent text expressing that content. This requires a model pre-trained on a task that forces it to (a) build rich global document representations and (b) generate coherent text.

### BART Pre-training (Denoising)

**The problem**: how do you pre-train an encoder-decoder model so that at fine-tuning time it already knows how to read and write?

**The core insight**: corrupt text in multiple ways, then train the model to reconstruct the original. A model that can fix deleted tokens, reorder shuffled sentences, and infill masked spans has learned everything summarization requires — reading for meaning, then generating from that meaning.

BART (Lewis et al., 2019) uses five corruption types:
- **Token masking** — replace tokens with `[MASK]`
- **Token deletion** — remove random tokens; model must infer positions
- **Text infilling** — replace arbitrary spans with a single `[MASK]` token
- **Sentence permutation** — shuffle sentence order
- **Document rotation** — rotate to start from a random token

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
```

### Fine-tuning

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
    predict_with_generate=True,
    evaluation_strategy="epoch",
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

**Key generation parameters**:
- `num_beams=4` — beam search width
- `length_penalty=2.0` — penalize short summaries (>1 favors longer)
- `no_repeat_ngram_size=3` — prevent repetitive output

**What breaks**: BART can hallucinate. It may generate a plausible-sounding but factually incorrect sentence because it optimizes fluency and ROUGE overlap with references, not factual correctness. Faithfulness evaluation requires separate metrics (FactCC, SummaC).

| Model | Architecture | Pre-training objective | Strength |
|-------|-------------|----------------------|----------|
| BART | Encoder-decoder | Denoising | Strong on news summarization |
| T5 | Encoder-decoder | Text-to-text on C4 | General-purpose |
| Pegasus | Encoder-decoder | Gap-sentence generation | Purpose-built for summarization |

---

## Extractive Summarization

### The problem

Abstractive models are slow, require GPU, and hallucinate. For applications where verbatim accuracy matters — legal documents, medical reports, earnings calls — you need a method that selects sentences rather than generating new ones.

### The core insight

Some sentences in a document are more central than others. A sentence referenced by many other sentences (high connectivity in a sentence similarity graph) is probably central to the document's meaning. Rank sentences by centrality, return the top-K.

---

### TextRank

**The core insight**: apply PageRank to a sentence similarity graph. Sentences that are similar to many other sentences in the document are likely to be the most salient. A sentence is "important" if it is connected to many other important sentences — the same recursive definition as PageRank.

```python
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def textrank_summarize(text: str, top_n: int = 3) -> str:
    import nltk
    sentences = nltk.sent_tokenize(text)

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform(sentences)
    sim_matrix = cosine_similarity(tfidf)

    n = len(sentences)
    scores = np.ones(n) / n
    damping = 0.85
    for _ in range(100):
        new_scores = (1 - damping) / n + damping * (sim_matrix @ scores)
        if np.allclose(scores, new_scores, atol=1e-6):
            break
        scores = new_scores

    ranked = np.argsort(scores)[::-1][:top_n]
    selected = sorted(ranked)
    return " ".join(sentences[i] for i in selected)
```

**What breaks**: TextRank scores sentences by their similarity to other sentences — not by their information value. A repeated boilerplate sentence ("This article was written by…") that appears many times scores highly if other sentences reference similar words.

---

### BERTSum

**The problem**: TF-IDF sentence representations miss semantic similarity. "The economy grew" and "GDP increased" have low TF-IDF cosine similarity but express the same idea.

**The core insight**: use BERT's contextual representations as sentence embeddings. Insert a `[CLS]` token before each sentence. Fine-tune a binary classifier on each `[CLS]` representation to predict whether the sentence belongs in the summary.

```
[CLS] sent_1 [SEP] [CLS] sent_2 [SEP] ... [CLS] sent_n [SEP]
```

```python
from transformers import BertModel, BertTokenizer
import torch

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert = BertModel.from_pretrained("bert-base-uncased")

def score_sentences(sentences: list[str]) -> list[float]:
    scores = []
    for sent in sentences:
        inputs = tokenizer(sent, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            out = bert(**inputs)
        cls_vec = out.last_hidden_state[:, 0, :]  # [CLS] token
        scores.append(cls_vec.norm().item())
    return scores
```

---

### Extractive vs Abstractive Trade-off

| | Extractive | Abstractive |
|--|------------|-------------|
| Faithfulness | High (verbatim) | Can hallucinate |
| Fluency | Choppy at boundaries | Natural, coherent |
| Compression | Limited by sentence granularity | Flexible |
| Speed | Fast | Slower (generation) |

---

## Summarization Metrics

### The problem

You have two summaries. One is grammatically perfect but misses the key facts. One is choppy but covers all the main points. How do you measure which is better automatically, without human raters?

### ROUGE

**The core insight**: if a good summary captures the main content of the reference, it should share n-grams with the reference. Measure n-gram recall (how much of the reference appears in the summary) and precision (how much of the summary appears in the reference).

```
ROUGE-N Recall    = |overlap of N-grams| / |N-grams in reference|
ROUGE-N Precision = |overlap of N-grams| / |N-grams in system output|
ROUGE-N F1        = harmonic mean
```

- **ROUGE-1**: unigram overlap (content coverage)
- **ROUGE-2**: bigram overlap (fluency, phrase preservation)
- **ROUGE-L**: longest common subsequence (order-sensitive)

```python
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(
    ["rouge1", "rouge2", "rougeL"],
    use_stemmer=True
)

reference  = "The cat sat on the mat near the window."
hypothesis = "A cat was sitting on the mat by the window."

scores = scorer.score(reference, hypothesis)
for key, val in scores.items():
    print(f"{key}: P={val.precision:.3f}  R={val.recall:.3f}  F={val.fmeasure:.3f}")
```

**What breaks**:

| Issue | Example |
|-------|---------|
| Penalizes valid paraphrases | "automobile" vs "car" — ROUGE=0, semantically correct |
| Rewards extractive copying | Copy source verbatim → high ROUGE, no compression |
| Ignores factual consistency | "Biden won in 2016" scores high if reference has "won in 2016" |
| No fluency signal | Shuffled n-grams can score well |

---

### BERTScore

**The problem**: ROUGE gives "automobile" and "car" a similarity of zero. A summary that uses synonyms throughout scores poorly even if it is semantically superior.

**The core insight**: compute cosine similarity between contextual BERT embeddings of hypothesis and reference tokens. Semantic equivalents that are lexically different still receive high similarity scores.

```python
from bert_score import score as bert_score

references  = ["The cat sat on the mat."]
hypotheses  = ["A cat was resting on the rug."]

P, R, F1 = bert_score(hypotheses, references, lang="en", verbose=False)
print(f"BERTScore F1: {F1.mean().item():.4f}")
```

**What breaks**: BERTScore requires a BERT forward pass per hypothesis-reference pair — 100-1000× slower than ROUGE. It also does not detect factual errors: a semantically plausible but wrong sentence can score highly.

---

### Factual Consistency Metrics

**The problem**: ROUGE and BERTScore both measure overlap with a reference summary. They do not check whether the generated summary is faithful to the *source document*. A summary can contradict the source while agreeing with the reference.

**The core insight**: use NLI to check whether each generated sentence is entailed by the source.

- **FactCC**: classify each generated sentence as supported/contradicted/neutral with respect to source
- **SummaC**: segment-level NLI — compute entailment probability between each source segment and each summary sentence; aggregate
- **QAEval**: generate QA pairs from the reference, check if the system summary answers them correctly

---

## Semantic Similarity and Sentence Embeddings

### The problem

You have 1 million product descriptions. A user queries "comfortable running shoes." You need to find the most semantically similar descriptions in milliseconds. The naive approach — run BERT's cross-attention over every query-description pair — would require 1 million forward passes per query.

### The core insight

Separate the encoding from the comparison. Encode all descriptions once and store their vectors. At query time, encode the query to a single vector, then do a dot-product search. BERT's cross-attention is replaced by cosine similarity.

**The problem**: plain BERT was not trained to produce good sentence-level vectors. Its CLS token was trained for NSP (next sentence prediction), which correlates weakly with semantic similarity. Fine-tuning specifically for sentence similarity is necessary.

---

### SBERT (Sentence-BERT)

**The core insight**: train a siamese network where two BERT encoders share weights, and the training objective is cosine similarity loss on known paraphrase pairs. After training, the model's mean-pooled output is a dense sentence embedding directly comparable by cosine similarity.

```
Sentence A → BERT → Mean Pool → u (768-dim)
Sentence B → BERT → Mean Pool → v (768-dim)

Training: minimize 1 − cos(u, v) for similar pairs
          maximize cos(u, v) penalty for dissimilar pairs
```

Mean pooling over token embeddings (not just CLS) proved more informative.

At inference: encode corpus once → store embeddings → query with single vector → ANN search. O(1) at query time after O(N) precomputation.

```python
from sentence_transformers import SentenceTransformer, util
import torch

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

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

corpus_embeddings = model.encode(corpus, convert_to_tensor=True)   # (8, 384)
query_embedding   = model.encode(query, convert_to_tensor=True)    # (384,)

hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=3)

for hit in hits[0]:
    print(f"  [{hit['score']:.4f}] {corpus[hit['corpus_id']]}")
# [0.6221] A man is riding a white horse on an enclosed piece of land.
# [0.5935] A man is riding a horse.
```

---

### Bi-encoder vs Cross-encoder

| | Bi-encoder (SBERT) | Cross-encoder |
|--|--------------------|-|
| Speed | Fast — pre-compute embeddings | Slow — full attention over pair |
| Accuracy | Slightly lower | Higher (full token interaction) |
| Use case | Retrieval, ANN search | Re-ranking top-K results |

**Production pattern**: bi-encoder for recall (retrieve top-100), cross-encoder to re-rank (select top-5). Near cross-encoder accuracy at bi-encoder scale.

**What breaks**: bi-encoders lose cross-token interaction between the two sentences. "The bank raised rates" and "The bank of the river" have similar SBERT embeddings because "bank" gets a context-averaged representation. Cross-encoders handle this because both sentences are concatenated before attention.

---

## Paraphrase Detection

### The problem

Two sentences: "How are you?" and "What is your state of being?" — are they paraphrases? "How are you?" and "What is the capital of France?" — clearly not. Surface word overlap is a poor signal: paraphrases can share zero words; non-paraphrases can share many (adversarial cases).

### The core insight

Paraphrase detection is a binary classification problem over sentence pairs. The two approaches differ in the tradeoff between speed and accuracy: bi-encoders are fast; cross-encoders are accurate.

```python
from sentence_transformers.cross_encoder import CrossEncoder

cross_encoder = CrossEncoder("cross-encoder/stsb-distilroberta-base")

pairs = [
    ("How are you?", "What is your state of being?"),
    ("How are you?", "What is the capital of France?"),
]

scores = cross_encoder.predict(pairs)
print(scores)
# [0.87, 0.03]  — high score means paraphrase
```

**Datasets**:
- **PAWS** — adversarial pairs constructed by word swapping + back-translation. Hard for surface-overlap models.
- **QQP** (Quora Question Pairs) — 400K question pairs, binary label
- **MRPC** — Microsoft Research Paraphrase Corpus

**What breaks**: PAWS was specifically designed to fool models that rely on word overlap. High BLEU/Jaccard similarity between sentences does not imply paraphrase; low similarity does not rule it out. Models must reason about meaning, not surface form.

---

## Natural Language Inference (NLI)

### The problem

A doctor reads a clinical note: "The patient had no fever, normal blood pressure, and was discharged the next day." A claim is: "The patient was seriously ill." Can a model determine whether the note supports, contradicts, or is silent on this claim?

This is a three-way classification problem that requires genuine language understanding — recognizing entailment, detecting contradiction, and knowing when inference is simply not possible.

### The core insight

Fine-tune an encoder transformer on premise-hypothesis pairs with labels {entailment, contradiction, neutral}. The model attends over both sentences jointly and learns to detect the logical relationship.

| Label | Meaning | Example |
|-------|---------|---------|
| **Entailment** | Premise implies hypothesis | P: "A dog is running." H: "An animal is moving." |
| **Contradiction** | Premise contradicts hypothesis | P: "A dog is running." H: "No animal is present." |
| **Neutral** | Cannot be inferred | P: "A dog is running." H: "The dog is brown." |

**The mechanics**:
```
[CLS] premise [SEP] hypothesis [SEP] → linear(768, 3) → softmax
```

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")
model = AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli")
# Labels: 0=contradiction, 1=neutral, 2=entailment

premise    = "A soccer game with multiple males playing."
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

**Datasets**:
- **SNLI** (570K pairs): crowd-sourced from image captions. Clean but narrow domain.
- **MultiNLI** (433K pairs): 10 genres (fiction, government, telephone…). More generalizable.
- **ANLI**: adversarially collected; much harder, three rounds of increasing difficulty.

**Applications**: fact-checking, QA answer verification, summarization faithfulness (FactCC), zero-shot classification.

**What breaks**: NLI models trained on SNLI/MultiNLI make systematic errors on domain-specific texts. "The patient was afebrile" entails "the patient had no fever" — but a model trained on news/fiction may not know "afebrile." Domain adaptation matters.

---

## Zero-Shot Classification with NLI

### The problem

You want to classify customer support tickets into 15 categories. Labeling 1,000 examples per category takes weeks. Can you classify without any labeled examples?

### The core insight

Phrase each candidate label as a hypothesis: "This text is about {label}." Use the entailment probability as the classification score. The NLI model was trained to detect when a premise implies a hypothesis — that is exactly what classification is.

```
Premise  = text to classify
Hypothesis = "This text is about politics."
Entailment probability → score for label "politics"
```

Run for all labels, softmax over entailment scores → zero-shot classifier.

```python
from transformers import pipeline

classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)

text = "The Federal Reserve raised interest rates for the third consecutive time, citing persistent inflation pressures."

labels = ["economics", "sports", "technology", "politics", "science"]

result = classifier(text, candidate_labels=labels)
for label, score in zip(result["labels"], result["scores"]):
    print(f"  {label:15s}: {score:.4f}")
# economics      : 0.8231
# politics       : 0.1124
```

**Multi-label mode** (each label independent — no softmax competition):
```python
result = classifier(text, candidate_labels=labels, multi_label=True)
```

**Custom hypothesis templates**:
```python
result = classifier(
    "The movie was an absolute masterpiece.",
    candidate_labels=["positive", "negative", "neutral"],
    hypothesis_template="The sentiment of this review is {}."
)
```

**What breaks**: hypothesis quality is the main lever. "This example is {label}" works generically; a poorly framed hypothesis can flip the prediction. Performance degrades on domain-specific categories the NLI model never saw during pre-training. Each label requires one full NLI forward pass — with 100 labels, inference is 100× slower than a trained classifier.

---

## Coreference Resolution

### The problem

An information extraction system reads: "Maria bought a house. She loves it. The property is near downtown." To extract the relation (Maria, owns, house), the system must know that "She" = "Maria" and "it" / "The property" = "a house." Without this, the extracted triple has two unresolved pronouns.

### The core insight

Find all noun phrases, names, and pronouns in the document. Score every pair of mentions for coreference likelihood. Cluster coreferent mentions into entity groups.

**The mechanics**:
1. **Mention detection**: find all candidate spans (noun phrases, pronouns, named entities)
2. **Mention pair scoring**: for each pair (i, j), score P(coreference) using span representations
3. **Clustering**: group spans with high pairwise coreference into entity clusters

```python
import spacy

nlp = spacy.load("en_coreference_web_trf")

doc = nlp("Maria bought a house. She loves it. The property is near downtown.")

for cluster_id, cluster in doc.spans.items():
    if cluster_id.startswith("coref_clusters_"):
        print(f"Cluster: {[span.text for span in cluster]}")
# Cluster: ['Maria', 'She']
# Cluster: ['a house', 'it', 'The property']
```

**Models**:
- **Stanford CoreNLP**: multi-pass sieve pipeline — deterministic rules first, then statistical models
- **SpanBERT (Lee et al., 2018)**: encode all spans, score pairs end-to-end. Current gold standard.

**What breaks**: coreference is hard for split-antecedent constructions ("John and Mary arrived. They were tired."), bridging references ("I bought a car. The engine was noisy."), and very long documents where the antecedent is hundreds of sentences earlier.

---

## Dependency Parsing

### The problem

You want to extract "who did what to whom" from a sentence. "The dog bit the man" and "The man bit the dog" have identical vocabulary and nearly identical TF-IDF representations — but completely different meanings. Surface features are insufficient. You need syntactic structure.

### The core insight

Assign a directed graph where each word points to its syntactic head, with a labeled relation type. This makes the predicate-argument structure explicit. Once you have "fox → jumps (nsubj)" and "dog → jumps (dobj via prep)", you know who jumped and what they jumped over.

```
nsubj     dobj
 ↑         ↑
"The cat ate the fish quickly"
                              ↑
                            advmod
```

**Universal Dependencies (UD)** defines a cross-lingual standard label set:

| Relation | Meaning | Example |
|----------|---------|---------|
| `nsubj` | Nominal subject | *cat* in "The **cat** eats" |
| `dobj` | Direct object | *fish* in "eats the **fish**" |
| `amod` | Adjectival modifier | *big* in "the **big** cat" |
| `advmod` | Adverbial modifier | *quickly* in "eats **quickly**" |
| `prep`/`case` | Prepositional attachment | *in* in "sits **in** the box" |

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The quick brown fox jumps over the lazy dog.")

for token in doc:
    print(f"{token.text:12s}  head={token.head.text:12s}  dep={token.dep_:10s}  pos={token.pos_}")

# fox          head=jumps         dep=nsubj      pos=NOUN
# jumps        head=jumps         dep=ROOT       pos=VERB
# dog          head=over          dep=pobj       pos=NOUN
```

**Visualize**:
```python
from spacy import displacy
displacy.serve(doc, style="dep")  # opens browser at localhost:5000
```

**What breaks**: parsers trained on newswire fail on social media ("lol u cant even") and code-mixed text. Prepositional phrase attachment is the oldest hard problem: "I saw the man with the telescope" — did you use the telescope to see, or did the man have a telescope?

---

## Relation Extraction

### The problem

You want to build a knowledge graph automatically from text: (Elon Musk, founded, SpaceX), (Marie Curie, born_in, Warsaw). Human annotation is expensive. You have millions of documents and no labels.

### The core insight

Three approaches, each solving a different bottleneck:
1. **Supervised (TACRED)**: when you have labeled data and a fixed schema
2. **Distant supervision**: when you have a knowledge base but no labeled text
3. **LLM prompting**: when you have neither, but can afford inference cost

---

### Supervised Approach — TACRED

TACRED (Zhang et al., 2017): 42 relation types + no_relation, ~100K examples from news/web. Mark entity spans with special tokens; classify the sentence.

```python
# Entity marking: wrap entities with special tokens
text = "[SUBJ-PERSON] Elon Musk [/SUBJ-PERSON] founded [OBJ-ORG] SpaceX [/OBJ-ORG] in 2002."

# Fine-tuned on TACRED: classifies into 42+1 relation types
```

**Standard pipeline**:
```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("Jeff Bezos founded Amazon in Bellevue, Washington.")
entities = [(ent.text, ent.label_, ent.start_char, ent.end_char) for ent in doc.ents]
# [('Jeff Bezos', 'PERSON', ...), ('Amazon', 'ORG', ...), ('Bellevue', 'GPE', ...)]

# For each entity pair → RE model → (subject, relation, object)
```

**What breaks**: supervised RE does not generalize to relation types not in the training schema. If TACRED has no "authored" relation and you want to extract (Shakespeare, authored, Hamlet), the model outputs no_relation.

---

### Distant Supervision

**The problem**: labeling 100K sentence-entity pairs requires months of annotation.

**The core insight**: use a knowledge base. If Wikidata says (Elon Musk, founded, SpaceX), any sentence containing both "Elon Musk" and "SpaceX" probably expresses this relation. Label automatically — no human annotation.

**What breaks**: "Elon Musk visited SpaceX headquarters" matches (Musk, founded, SpaceX) by entity co-occurrence but expresses no such relation. **Multi-instance learning** mitigates this by aggregating evidence over all sentences for an entity pair rather than treating each sentence independently.

---

### LLM-Based Extraction

**The problem**: distant supervision is noisy; supervised RE is schema-constrained. What if you want open-schema extraction from a new domain with no labeled data?

**The core insight**: LLMs have absorbed relation knowledge from pretraining. Prompt them with the text and ask for triples directly.

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
#   {"subject": "Marie Curie", "relation": "born_in",   "object": "Warsaw"},
#   {"subject": "Marie Curie", "relation": "worked_at", "object": "University of Paris"},
#   {"subject": "Marie Curie", "relation": "won",       "object": "Nobel Prize"},
# ]
```

| | Supervised (TACRED) | Distant Supervision | LLM-based |
|--|---------------------|---------------------|-----------|
| Accuracy | High (in-domain) | Noisy | High (flexible) |
| Schema | Fixed | Fixed | Open / flexible |
| Data need | Labeled pairs | KB + corpus | None (zero-shot) |
| Speed | Fast inference | Fast inference | Slow + costly |
| Generalization | Poor (new relations) | Poor | Good |

**What breaks**: LLM extraction is inconsistent — the same text may produce different triples across calls. Relation names vary ("born_in", "birthplace", "was born in"). Post-processing and schema normalization are required for knowledge graph population.

---

## Key Points

**Abstractive vs Extractive Summarization**
- Extractive: score and select sentences; fast, faithful, but grammatically choppy at boundaries
- Abstractive: generate new text; more fluent and flexible, but can hallucinate
- BART denoising: corrupts text (masking, deletion, permutation, shuffling) and trains decoder to reconstruct — gives both strong document encoding and fluent generation

**ROUGE limitations**
- ROUGE rewards n-gram surface overlap with a reference — penalizes valid paraphrases, rewards verbatim copying
- High ROUGE does not imply factual correctness or fluency
- Complement with BERTScore (semantic similarity), FactCC/SummaC (faithfulness to source)

**SBERT vs plain BERT for similarity**
- Plain BERT: O(N²) cross-encoder forward passes for N-sentence comparison
- SBERT: encode once → cached embeddings → cosine similarity search in O(1) at query time
- Mean pooling over token embeddings outperforms CLS-only
- Production pattern: SBERT for recall (retrieve top-K), cross-encoder for re-ranking (precision)

**NLI as universal tool**
- NLI models (MultiNLI fine-tuned) power zero-shot classification via hypothesis templates
- "This text is about {label}" — entailment probability is the classification score
- Also used for fact-checking, summarization faithfulness (FactCC), paraphrase detection

**Coreference resolution**
- Three stages: mention detection → pairwise scoring → clustering
- SpanBERT end-to-end is the current standard
- Critical upstream of relation extraction: "He acquired it" cannot be resolved without knowing who "he" is

**Dependency parsing**
- Every word has one head (except root); edges are labeled (nsubj, dobj, amod…)
- Universal Dependencies enables cross-lingual consistency
- Used upstream of relation extraction, semantic role labeling, grammar correction

**Relation extraction**
- Supervised (TACRED): fixed schema, high precision, does not generalize to unseen relations
- Distant supervision: scalable but noisy; multi-instance learning helps
- LLM-based: zero-shot, open-schema, flexible — best for prototyping or low-resource scenarios
- Full pipeline: NER → entity pair enumeration → RE model → knowledge graph population
