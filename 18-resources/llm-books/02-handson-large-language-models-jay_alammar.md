---
module: Llms
topic: Books
subtopic: Handson Large Language Models Jay Alammar
status: unread
tags: [llms, ml, books-handson-large-language-m]
---
# Hands-On Large Language Models — Jay Alammar
## First-Principles Notes: Problem First, Concept Second

---

## Chapter 1: What LLMs Are and Why the Distinction Matters

### The Problem
"LLM" is used to describe both systems that generate text and systems that understand text. Using GPT-4 for document classification wastes 99% of its capability. Using BERT to write a blog post produces nothing. The wrong model for the task is not a performance problem — it is a category error. You need a framework for choosing before you build.

### The Core Insight
LLMs split cleanly into two architectures with different purposes. Conflating them leads to wrong tool selection.

**Representation Models (Encoder-Only)**: Convert text to vectors. Do not generate text. Used for classification, clustering, semantic search. BERT is the canonical example. "Flashy" at nothing — they are industrial backend components.

**Generative Models (Decoder-Only)**: Predict the next token in a sequence. Used for chatbots, completion, content creation. GPT series. All the public attention goes here.

The word "Large" in LLM is not a meaningful threshold. A smaller, well-optimized model that captures language effectively is still an LLM. A massive model that is poorly trained on the wrong data is not necessarily better. Capability, not parameter count, is the definition.

### The Mechanics
API interaction standard: a list of message dicts with `role` and `content` fields.
```python
[{"role": "user", "content": "Create a funny joke about chickens."}]
```
This format is universal across modern chat models and should be adopted immediately as the standard interaction protocol.

**Proprietary vs. Open Source**:
- Proprietary (OpenAI, Cohere): API access, no hardware required, black box, usage costs, privacy concerns.
- Open Source (Hugging Face, Llama): Requires GPU, full access to weights and internals, privacy, customization.

### What Breaks
- Using a generative model for classification when a BERT-class model would be 10× cheaper and faster.
- Dismissing a model as "not an LLM" because it is small — the relevant question is whether it effectively captures language for the target task.
- Relying on proprietary APIs when data cannot leave the organization (healthcare, legal, finance).

---

## Chapter 2: Tokenization and Embeddings

### The Problem
Computers cannot process text. They process numbers. The naive solution — count word frequencies (Bag-of-Words) — ignores meaning. "The bank was steep" and "The bank was solvent" look identical if you only count the word "bank." Any system that ignores semantics fails on any task requiring understanding.

### The Core Insight
Meaning is defined by context. Words that appear in similar contexts have similar meanings. A neural network trained to predict which words co-occur learns to represent similar words as nearby vectors in a high-dimensional space. Distance in that space approximates semantic similarity.

### The Mechanics
**Bag-of-Words**: Represent text as a frequency count per word. Fast. Ignores order and context. Fails on synonyms, polysemy, and any semantically nuanced task.

**Word2Vec (2013)**: A neural network trained on massive text corpora to predict neighboring words. Result: each word maps to a dense vector (e.g., 300 dimensions). "King" and "Queen" land near each other. "Dog" and "canine" land near each other. The famous arithmetic: `vector("King") - vector("Man") + vector("Woman") ≈ vector("Queen")`.

Training mechanism: for a given word, predict whether another word is its neighbor in a sentence. Words with similar neighbors converge to similar vectors.

**Limitation**: Word2Vec produces *static* embeddings. "Bank" has one vector regardless of whether it appears in a financial or geographical context. Modern Transformers produce *contextual* embeddings — the same word gets a different vector depending on surrounding context.

**Tokenization**: Text must be split into *tokens* before embedding. Tokenizers do not split on spaces alone.

- Subword tokenization (BERT): "CAPITALIZATION" → `['CA', '##PI', '##TA', '##L', '##I', '##Z', '##AT', '##ION']`. Rare or compound words are decomposed into known subword units.
- Special tokens: `[CLS]` (start of sequence), `[SEP]` (sequence separator), `[PAD]` (batch padding), `[UNK]` (out-of-vocabulary fallback).
- Vocabulary sizes: typically 30,000–100,000 tokens.
- Capitalization matters: the same word in different cases may produce different token IDs.

### What Breaks
- Bag-of-Words for any task requiring semantic understanding — it is a frequency table, not a meaning representation.
- Ignoring tokenization when debugging: if a model performs badly on rare technical terms, inspect how those terms are tokenized first.
- Using static Word2Vec embeddings for tasks where word meaning shifts with context (sentiment, ambiguity resolution).

---

## Chapter 3: The Transformer Architecture

### The Problem
Static embeddings (Chapter 2) give every word one fixed meaning. Language is not like that — "I went to the bank" means something completely different depending on whether the previous sentence was about rivers or loans. To handle context, a model must be able to look at the *entire* surrounding sequence when computing the representation of any given word.

### The Core Insight
Let every token in a sequence attend to every other token. The representation of a word is not fixed — it is computed as a weighted sum over all other words in the sequence, where the weights are learned. This is attention.

### The Mechanics
**The forward pass pipeline**:
```
Input text → Tokenizer → Token IDs → Embedding Layer → Stack of Transformer Blocks → LM Head → Probability distribution over vocabulary
```

**The LM Head**: The final linear layer that maps the last hidden state to a vocabulary-sized vector. Softmax converts it to a probability distribution. The highest-probability token is selected as the next output.

**Scaled Dot-Product Attention**:
```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V
```
- Q (Query): what this token is looking for.
- K (Key): what each other token offers.
- V (Value): the actual content each token contributes.
- Division by √d_k: as d_k grows, dot products grow in magnitude, pushing softmax into near-zero-gradient regions. Scaling by √d_k keeps variance around 1.

**Positional Encodings**: Attention is order-agnostic by construction. A sentence and its scrambled version have the same attention graph unless position is injected explicitly. Positional encodings encode the index of each token.

**RoPE (Rotary Positional Embeddings)**: Applied *inside* the attention calculation to Q and K matrices — not at the input embedding layer. This is a common misconception. RoPE encodes *relative* position using geometric rotation, making it more effective for long sequences. Used in Llama 2 and later models.

**KV Cache**: During autoregressive generation, each new token requires attention over all previous tokens. Instead of recomputing the Key and Value matrices for every previous token at each step, they are cached after first computation. Only the new token's Query is computed fresh. Cache size grows linearly with sequence length × batch size — this is why long context is memory-intensive.

**Sparse Attention**: Full attention costs O(n²) in sequence length. Sparse attention limits each token to attending only to a local window of neighbors. Faster, but risks losing long-range dependencies. GPT-3's solution: alternate between sparse and full attention blocks.

### What Breaks
- Long sequences: attention is the most computationally expensive part of the forward pass. Processing a 100,000-token document with full attention is prohibitively expensive.
- KV cache exhaustion: for long conversations or large batch sizes, the KV cache fills GPU VRAM. This is why real-time chatbots have context limits in production.
- Abrupt output cutoff: when `max_new_tokens` is hit, generation stops mid-sentence. The model is not confused — it hit the limit you set.

---

## Chapter 4: Classification

### The Problem
You have a corpus of text and a labeling task (spam/not-spam, positive/negative, topic category). You need a model that predicts the correct label reliably. Two approaches exist, and they have dramatically different cost, accuracy, and data requirements. Choosing the wrong one wastes compute or produces inferior results.

### The Core Insight
Classification can be framed as either a structured prediction problem (give me a label) or a generation problem (generate the correct word). Representation models (BERT-class) are purpose-built for the first framing. Generative models (GPT-class) can do the second but with different trade-offs.

### The Mechanics
**Representation Approach (Specialist)**:

Option A — Task-specific pretrained model: find a model on Hugging Face already fine-tuned for your domain (e.g., `twitter-roberta-base-sentiment`). Zero additional training required.

Option B — Embeddings + Logistic Regression:
1. Freeze the BERT model (no gradient updates).
2. Pass training examples through BERT to extract fixed embeddings.
3. Train a Logistic Regression classifier on those embeddings.
Result: ~85% F1 on standard benchmarks with minimal compute.

Option C — Zero-shot embedding classification:
1. Embed the class labels themselves ("positive", "negative").
2. At inference, embed the input text.
3. Assign the label whose embedding is nearest (cosine similarity).
No labeled data required. Achieves ~78% F1 on the Rotten Tomatoes benchmark — meaningfully weaker than the trained Logistic Regression (85%) but requires zero examples.

**Generative Approach (Generalist)**:
Use prompt engineering to force the LLM to output a structured classification signal:
> "If the review is positive return 1, if negative return 0. Do not give any other answer."

Flexible (no training data, no fine-tuning), but expensive per query. Using GPT-4 to output a binary digit is computationally wasteful for high-volume tasks.

**T5 (Text-to-Text Transfer Transformer)**: An encoder-decoder architecture that reformulates every NLP task as a text generation task. "Classify this as positive or negative" → model generates "positive". A bridge between the two approaches.

**Exponential Backoff**: When calling proprietary APIs, rate limits are a production reality. Retry with geometrically increasing wait times between attempts: wait 1s, then 2s, then 4s, then 8s.

### What Breaks
- Using GPT-4 to classify 10 million emails: the cost is unjustifiable when a 110M-parameter BERT model achieves similar accuracy at 1/1000th the cost.
- Failing to constrain generative model output: a generative model prompted to classify will sometimes produce a sentence of explanation instead of the label. Enforce strict output format.
- Zero-shot embedding classification on domain-specific jargon: the embedding space may not meaningfully separate technical categories that a general model has not seen.

---

## Chapter 5: Clustering and Topic Modeling

### The Problem
You have 45,000 documents and no labels. Traditional supervised classification requires labels you do not have. Even if you had labels, defining the categories in advance assumes you already know what themes exist — which defeats the purpose of exploration. How do you discover hidden structure in a corpus?

### The Core Insight
Semantic similarity in embedding space corresponds to topical relatedness. Documents that discuss the same subject will embed near each other. Clustering algorithms that operate on embedding vectors can group documents into thematic clusters without any labels. The remaining challenge: the clusters are unlabeled — you need to extract what each cluster is *about*.

### The Mechanics
**The Text Clustering Pipeline**:
```
Documents → Embeddings → Dimensionality Reduction (UMAP) → Density Clustering (HDBSCAN) → Cluster label extraction
```

**Why UMAP before clustering**: Raw embeddings are high-dimensional (768 or more dimensions). Distance metrics lose meaning in high-dimensional space (the curse of dimensionality). UMAP projects embeddings into 2D or 5D while preserving local structure.

**Why HDBSCAN over K-Means**:
- K-Means requires you to specify the number of clusters in advance.
- HDBSCAN finds arbitrarily shaped clusters based on density.
- Critically: HDBSCAN assigns label `-1` to points that don't belong to any dense cluster. These noise points are excluded from topic descriptions rather than being forced into the nearest cluster.

**BERTopic**: Extends the pipeline with a representation step.

**c-TF-IDF (Class-based TF-IDF)**: After clustering, treat all documents in a cluster as a single mega-document. Compute TF-IDF across clusters rather than across individual documents. Words that appear frequently in one cluster but rarely in others become that cluster's keywords. This converts a pile of vectors into a human-readable topic label like "Machine Learning" or "Healthcare."

**MMR (Maximal Marginal Relevance)**: Prevents redundant keywords. Without MMR, a "Summarization" topic might be described as: `summarization | summaries | summary | summarize`. With MMR: `summarization | document | extractive | rouge`. Adds diversity by penalizing keywords that are too similar to already-selected keywords.

### What Breaks
- Embedding quality ceiling: if the underlying model does not understand your domain's jargon (legal, biomedical), semantically related documents will not embed near each other. Domain-specific embedding models are required.
- UMAP stochasticity: UMAP results vary across runs. Cluster assignments can change between executions with no change in input data. Set a random seed for reproducibility.
- Trusting keyword lists blindly: inspect actual documents within each cluster. A keyword like "patient" could describe a medical cluster or, coincidentally, a cluster about customer service.

---

## Chapter 6: Semantic Search and RAG

### The Problem
Keyword search (Ctrl+F, BM25) requires the exact word to be present in the document. "Can dogs eat grapes?" does not match "Grapes are toxic to canines" by keyword. Worse, a system that returns the closest keyword match regardless of relevance produces nonsense for queries that have no good match. And a generative model asked about facts it does not know invents plausible but wrong answers.

### The Core Insight
Search is a geometric problem. Embed the query; embed all documents; the answer is in the neighborhood of the query in vector space. Combine this with generation: retrieve relevant facts first, then generate an answer grounded in those facts. Hallucination is a knowledge gap — retrieval fills the gap before generation.

### The Mechanics
**Three tiers of retrieval sophistication**:

**1. Dense Retrieval (Fast)**:
- Embed query and all documents independently using a Bi-Encoder.
- At query time: embed the query, find the K nearest document embeddings by cosine similarity.
- Documents are pre-embedded and pre-indexed — query time is just a vector lookup.
- Scales to millions of documents with ANN (Approximate Nearest Neighbor) indexes: FAISS (Facebook AI Similarity Search), HNSW.
- Weakness: query and answer may not be near each other in embedding space because they look different ("Who is the CEO?" ≠ "Satya Nadella").

**2. Reranking (Accurate)**:
- Cross-Encoder: takes (query, document) as a *pair*, processes them together, outputs a relevance score.
- Far more accurate than Bi-Encoder because it can model the relationship between query and document.
- Cannot scale to millions of documents — must process each pair at query time.
- Used as a second pass: Dense Retrieval returns top-100 candidates; Reranker scores each and returns top-10.

**3. The Production Pipeline**:
```
Query → Bi-Encoder → Top-100 candidates → Cross-Encoder Reranker → Top-10 → LLM generates grounded answer
```

**Hybrid Search**: Combine dense retrieval (semantic) with BM25 (keyword). Required for queries involving exact matches: product serial numbers, error codes, rare proper nouns. Purely semantic search fails on `0x404` or `SKU-83291`.

**RAG (Retrieval-Augmented Generation)**:
```
Query → Retrieve relevant chunks → Inject into prompt → "Answer only using the provided text" → LLM responds
```
RAG reduces hallucination by giving the model a *source of truth* before it generates. It also enables citations: the model can point to the specific retrieved chunk it used.

**Similarity threshold**: If the nearest document is still far in vector space, return "no results found" rather than the nearest irrelevant document. A threshold is mandatory for production quality.

### What Breaks
- Bi-Encoder recall ceiling: queries phrased unusually may not embed near the correct answer. Reranking is not optional for high-stakes retrieval.
- Chunk boundary artifacts: if the answer spans two non-adjacent chunks, the model sees half the context from each and may miss the connection.
- Hallucination despite RAG: if all retrieved chunks are irrelevant, the model may still generate a confident wrong answer. Enforce "If the provided text does not contain the answer, say so."
- Dense retrieval on exact-match queries: semantic embeddings of "order #83291" and "order #83292" are essentially identical. Keyword search is required for exact matches.

---

## Chapter 7: Advanced Text Generation — Chains and Memory

### The Problem
A single LLM call is stateless and bounded. It cannot write a consistent 10,000-word document in one pass. It forgets everything between API calls. It fails at multi-step math because it jumps to answers without checking intermediate steps. These are not model limitations — they are limitations of how the model is being called.

### The Core Insight
Complex generation problems decompose into sequences of simpler generation problems. Each step's output becomes the next step's input. Memory is not a property of the model — it is a property of the context you construct and pass.

### The Mechanics
**Chain-of-Thought (CoT)**: Append "Let's think step by step" to any math or logic prompt. The model generates intermediate reasoning steps as text. Each generated step becomes input to the next step in the same generation pass. The model cannot skip a step without generating it explicitly, forcing it to commit to correct intermediate values.

Zero-Shot CoT: no examples needed — the magic phrase alone triggers structured reasoning. Few-Shot CoT: provide worked examples of the step-by-step format before the question.

**Sequential Chains**:
```
Step 1: "Write an outline for an article about X" → {outline}
Step 2: "Write the introduction based on this outline: {outline}" → {intro}
Step 3: "Write section 1 based on: outline={outline}, intro={intro}" → {section_1}
```
Each step can use only what it needs — do not pass everything everywhere.

**Memory architectures** (LLMs are stateless; memory is injected context):

`ConversationBufferMemory`: Store every turn of conversation verbatim. Inject the full history into each new prompt.
- Pros: maximum fidelity, nothing forgotten.
- Cons: context window fills fast; cost and latency grow linearly with conversation length.

`ConversationSummaryMemory`: Periodically summarize older turns with an LLM call. Inject the summary instead of raw history.
- Pros: bounded context size, saves tokens.
- Cons: summary call adds latency; summarization loses nuance; errors in the summary compound over time.

Sliding window memory: keep only the last N turns. Simple, predictable, lossy.

### What Breaks
- Single-pass complex generation: long-form coherent content requires sequential planning steps. One-shot prompts drift and lose consistency.
- Memory without pruning: `ConversationBufferMemory` for long sessions will eventually hit the context window limit and crash the application.
- Summary compounding errors: if the summary in turn 20 incorrectly summarizes turn 10, every subsequent turn inherits that error. Summaries must be validated or kept append-only.
- CoT without commitment: some models generate reasoning steps and then ignore them when formulating the final answer. Force the model to reference its own chain explicitly in the final answer.

---

## Chapter 8: Semantic Search and RAG (Deep Dive)

### The Problem
The shift from keyword search to semantic search was among the most impactful applications of Transformers — Google and Bing integrated BERT-class models into their search engines as early as 2018, marking a step-change in search quality. But practitioners building their own systems often implement only half the pipeline and wonder why results are mediocre.

*Note: Chapter 8 deepens the same pipeline introduced in Chapter 6. These notes focus on the distinctions and additional mechanisms.*

### The Core Insight
The query-answer semantic gap is a real problem. A question and its answer are not semantically similar in a general embedding space. A system trained on generic text will embed "What did Marie Curie discover?" far from "Radium and Polonium" because questions and factual sentences look different structurally. The solution is either a retrieval model specifically trained on question-answer pairs, or a reranker that sees both query and candidate together.

### The Mechanics
**Bi-Encoder retrieval** at industrial scale:
- Documents are embedded once, offline, and stored in a FAISS index.
- At query time: single embedding computation + nearest-neighbor lookup. Millisecond latency at millions-of-document scale.
- FAISS supports both exact search (IVF) and approximate search (HNSW). Approximate is faster; exact is more accurate.

**Cross-Encoder reranking**:
- Input: `[CLS] query [SEP] document [SEP]`
- Output: a single relevance score.
- The model sees query-document interaction explicitly, not just their independent embeddings.
- "Vastly improved results" compared to Bi-Encoder alone, at the cost of O(candidates) inference calls per query.

**FAISS index construction**:
```python
import faiss
index = faiss.IndexFlatL2(embedding_dim)
index.add(document_embeddings)  # all documents
scores, indices = index.search(query_embedding, k=100)  # top-100 candidates
```

**RAG with Cohere**:
```python
results = co.rerank(query=query, documents=candidates, top_n=10)
context = "\n".join([r.document["text"] for r in results])
answer = llm.generate(f"Answer using only this context:\n{context}\n\nQuestion: {query}")
```

### What Breaks
- Generic Bi-Encoders on domain-specific corpora: a general embedding model has no notion of your internal taxonomy. Fine-tuning the retrieval model on domain-specific Q-A pairs is necessary for high retrieval recall.
- Indexing without chunking: embedding an entire 50-page document as a single vector produces a vector that averages over all topics in the document. The query "what is the refund policy?" will not find the correct paragraph if it is averaged with 49 other pages.
- Returning results past the relevance threshold: for uncommon queries with no good match in the corpus, the nearest result may still be wrong. A distance cutoff is mandatory.

---

## Chapter 9: Multimodal Large Language Models

### The Problem
Humans communicate through images, documents, charts, and diagrams, not just text. A model that processes only text cannot read a contract's table layout, interpret a medical scan, or understand the relationship between a chart and its caption. OCR + LLM pipelines lose the spatial and visual context. A purely text-based system is blind to a large fraction of real-world information.

### The Core Insight
If images can be tokenized into sequences of vectors — "visual words" — they can be fed into a Transformer that already knows how to process sequences. The bridge is a projection layer that maps pixel-space representations into the same embedding space as text tokens. The model then processes image patches and text tokens in the same forward pass.

### The Mechanics
**Vision Transformer (ViT)**:
1. Take an image (e.g., 224×224 pixels).
2. Divide it into a grid of non-overlapping patches (e.g., 16×16 patches, each 14×14 pixels).
3. Flatten each patch into a vector.
4. Add positional embeddings to encode patch location.
5. Feed the sequence of patch vectors through a standard Transformer.

Each image patch is analogous to a text token. A 224×224 image divided into 16×16 patches becomes a sequence of 196 "visual tokens."

**Projection Layer**: ViT produces visual embeddings in a different space than text embeddings. A learned linear projection maps visual embeddings into the text embedding space. This allows the LLM to process `[image tokens] [text tokens]` in a single unified forward pass.

**CLIP (Contrastive Language-Image Pre-Training)**:
- Trained on millions of (image, caption) pairs.
- Loss: maximize similarity between matched image-text pairs, minimize similarity between mismatched pairs.
- Result: image embeddings and text embeddings are aligned — `embed("a photo of a dog")` is near the embedding of an actual dog photo.
- Application: zero-shot image classification. Classify by finding which text label embedding is nearest to the image embedding. No image-specific training required.

**Zero-shot image classification with CLIP**:
```python
image_embedding = clip.encode_image(image)
label_embeddings = [clip.encode_text("a photo of a cat"), clip.encode_text("a photo of a dog")]
predicted_label = argmax(cosine_similarity(image_embedding, label_embeddings))
```

### What Breaks
- Context window consumption: image patches are tokens. A 224×224 image at 16×16 patches = 196 tokens, filling roughly 2-3% of a 8k context window per image. High-resolution images consume context rapidly.
- Resolution sensitivity: ViT models expect specific input sizes (224×224 is standard). Images that are not resized to this specification before input produce degraded results or errors.
- CLIP on domain-specific imagery: CLIP was trained on internet images. Medical imaging, satellite imagery, or industrial inspection images may not embed correctly without domain-specific fine-tuning.
- Layout vs. content: multimodal models can read text within images, but spatial understanding of document layout (tables, forms, flowcharts) requires architectures specifically designed for document understanding, not just general ViT + LLM.

---

## Chapter 10: Creating Text Embedding Models

### The Problem
Pre-trained general embedding models (BERT, OpenAI ada-002) define similarity as generic semantic relatedness. "I love this movie" and "I hate this movie" embed near each other because both are about movies. For a sentiment-analysis system, this is exactly wrong — these sentences should be as far apart as possible in the embedding space. A general embedding model optimized for topic similarity cannot serve applications that require sentiment, style, or domain-specific similarity.

### The Core Insight
The embedding space is not fixed. It is a learned representation that can be reshaped through fine-tuning. By training the model on examples of what "similar" means for your specific task, you reshape vector space so that your definition of proximity is enforced.

### The Mechanics
**Contrastive Learning** is the primary training technique:

Goal: Similar documents → close in vector space. Dissimilar documents → far in vector space.

Training data structure:
- **Positive pairs**: (document A, document B) where A and B are semantically equivalent under your definition of similarity.
- **Negative pairs**: (document A, document C) where A and C are not similar.
- **Hard negatives**: (document A, document D) where D *looks* similar (shared keywords) but is actually wrong. Hard negatives are far more informative than random negatives.

Training loss: minimize distance for positive pairs, maximize for negative pairs. The mathematical core is the contrastive loss or triplet loss.

**Contrastive Explanation analogy**: A reporter asks a bank robber "Why did you rob a bank?" The robber says "Because that's where the money is." The robber answered "Why a bank instead of a bakery?" not "Why robbery instead of employment?" Without a *contrast* (the negative case), the question is ambiguous. Models face the same ambiguity during training — they need negatives to understand *what makes things different*.

**What fine-tuning changes**: Takes a general model (clusters by topic) and reshapes it. Example result:
- Before: "I love this movie" ≈ "I hate this movie" (both are movie reviews)
- After: "I love this movie" ≈ "This was wonderful" and far from "I hate this movie"

**Sentence Transformers library**: The standard tool for bi-encoder fine-tuning. Define training pairs, specify loss function, train with `model.fit()`.

### What Breaks
- Catastrophic forgetting: aggressive fine-tuning can cause the model to lose general language understanding while becoming expert at the target task. Use a low learning rate and mix in general-domain examples.
- Low-quality negatives: random negatives are easy to separate. The model learns trivially and fails on hard cases. Curate hard negatives explicitly.
- Misdefining similarity: if you want to cluster by author style but your training pairs reflect topic similarity, the fine-tuned model will cluster by topic. The model learns what you show it, not what you intend.
- Evaluation against the wrong metric: if you fine-tune for sentiment similarity but evaluate with a topic-similarity benchmark, the model looks like it got worse. Define your evaluation metric before defining your training pairs.

---

## Chapter 11: Fine-Tuning Representation Models for Classification

### The Problem
Using a frozen pre-trained model as a feature extractor (Chapter 4's embedding + Logistic Regression approach) is fast but suboptimal. The internal representations of a frozen BERT model were learned on generic text. Your domain — movie reviews, medical records, legal contracts — has vocabulary, style, and semantic relationships that the pre-trained model has not specifically optimized for. The model's internal weights are wrong for your task.

### The Core Insight
Pre-training gives a model general language understanding. Fine-tuning gives it task-specific language understanding. By allowing gradient updates to propagate through the entire model during task-specific training, every layer can adapt its representations to the target domain. The result is measurably better performance on the specific task.

### The Mechanics
**Full Fine-Tuning workflow**:
1. Load pre-trained BERT.
2. Replace the final layer with a classification head: a linear layer projecting from hidden size (768) to number of classes (e.g., 2 for binary sentiment).
3. Unfreeze all layers — make all weights trainable.
4. Train on labeled data with a small learning rate (1e-5 to 3e-5). The pre-trained weights are the starting point; the task-specific data pulls them toward the optimum for your task.

**Frozen vs. fine-tuned comparison** (Rotten Tomatoes benchmark):
- Frozen BERT + Logistic Regression (Chapter 4): F1 = 0.80
- Full fine-tuning (this chapter): F1 = 0.85

5 points of F1 from allowing the model to update its internal representations.

**Layer freezing strategies**:
- Freeze bottom N layers (basic grammar features — these generalize well), fine-tune top layers (complex semantics — these are task-specific).
- Reduces compute while preserving most accuracy gain.

**SetFit (Few-Shot Classification)**: When labeled data is scarce, use contrastive learning on sentence embeddings to create a compact classifier from very few examples. Avoids the data requirements of full fine-tuning.

**Token Classification (NER)**: Extends classification from document level ("is this email spam?") to token level ("is this word a PERSON, LOCATION, or DATE?"). Each token position gets its own classification head output. Useful for information extraction from structured text.

**Hugging Face Trainer API**:
```python
trainer = Trainer(
    model=model,
    args=TrainingArguments(output_dir="./results", num_train_epochs=3),
    train_dataset=train_data,
    eval_dataset=eval_data,
)
trainer.train()
```

### What Breaks
- Overfitting on small datasets: fine-tuning a 110M-parameter model on 500 examples will memorize the training data. Monitor validation loss; stop training when it starts increasing.
- Catastrophic forgetting: a model fine-tuned too aggressively on task-specific data loses general language understanding. Regularization (low learning rate, dropout) mitigates this.
- Deployment proliferation: full fine-tuning produces a separate 12GB model file per task. Storing a fine-tuned BERT for 20 classification tasks = 240GB. Parameter-efficient fine-tuning (LoRA) addresses this but is not covered in depth in this chapter.

---

## Chapter 12: Fine-Tuning Generation Models

### The Problem
A base language model trained on internet text will, when given the prompt "Write a poem about autumn," continue with "...and other creative writing prompts are:" — it treats the instruction as text to be continued, not as a command to be followed. Base models are trained to predict the next token, not to be assistants. The gap between "predicts next token competently" and "follows instructions helpfully" requires a dedicated training stage.

### The Core Insight
There are three distinct problems, and they require three distinct interventions:
1. The model does not know the instruction-following format → Supervised Fine-Tuning (SFT).
2. The model cannot be run on consumer hardware for fine-tuning → Parameter-efficient adaptation (LoRA/QLoRA).
3. The model follows instructions but not in alignment with human preferences → Preference optimization (DPO).

### The Mechanics
**Base Model vs. Instruct Model**:
- Base Model: trained on raw text, predicts next token. Useful for researchers, not users.
- Instruct Model: fine-tuned to follow instructions. The product users interact with.

**Stage 1: Supervised Fine-Tuning (SFT)**
Train on `(instruction, ideal_response)` pairs. The model learns: "When I see input structured like a user instruction, I should output a structured helpful response."

Data quality dominates data quantity. The "LIMA" dataset principle: 1,000 high-quality curated examples outperform 50,000 generated or low-quality examples.

**Stage 2: Parameter-Efficient Fine-Tuning (PEFT) — LoRA**

The problem with full fine-tuning of 70B models: requires ~140GB of VRAM to store weights + gradients + optimizer states. No consumer hardware can do this.

LoRA (Low-Rank Adaptation):
- Freeze all original model weights.
- For each target weight matrix W (usually Q and V in attention layers), add two small matrices: W = W_0 + BA, where B is (d × r) and A is (r × d), with r << d.
- Only B and A are trained. If r = 8 and d = 4096, you train 2 × 4096 × 8 = 65,536 parameters instead of 4096 × 4096 = 16,777,216.
- Reduces trainable parameters by 99%+ while achieving performance close to full fine-tuning.

QLoRA (LoRA + Quantization):
- Quantize the frozen base model to 4-bit precision (reduces model size by 4×).
- Apply LoRA adapters in 16-bit precision.
- Fine-tune a 70B model on a single 48GB GPU.

**Stage 3: Preference Alignment**

SFT teaches format and helpfulness. It does not teach safety, tone, or the ability to refuse inappropriate requests.

RLHF (old approach):
1. Generate multiple model responses to the same prompt.
2. Human raters rank the responses.
3. Train a Reward Model on the rankings.
4. Use PPO (Proximal Policy Optimization) to optimize the LLM to maximize Reward Model scores.
Complex, unstable, requires managing three separate models.

DPO (Direct Preference Optimization, new approach):
- Input: pairs of (chosen response, rejected response) for the same prompt.
- Loss: directly optimize the model to prefer "chosen" over "rejected."
- No separate Reward Model. No PPO. One stable training loop.
- Result: equivalent alignment quality, significantly simpler implementation.

### What Breaks
- SFT on low-quality data: the model learns to follow the format of the training data, including its errors and biases. Garbage in, garbage out — at massive scale.
- LoRA rank selection: too small a rank (r=1, r=2) underfits — the adapter cannot represent the required adaptations. Too large a rank approaches full fine-tuning. r=8 or r=16 are common starting points.
- DPO over-optimization (Alignment Tax): heavily aligning the model to human preference data can reduce output diversity. The model converges to "safe and agreeable" responses and loses capability on tasks requiring creative or unconventional reasoning. Heavy alignment can measurably degrade coding and reasoning benchmark scores.
- Base model as the wrong starting point: fine-tuning a base model on instruction data without first applying SFT from the provider's pipeline produces erratic behavior. Always start SFT from the base model, not from a partially-aligned checkpoint you cannot inspect.

## Flashcards

**Proprietary (OpenAI, Cohere)?** #flashcard
API access, no hardware required, black box, usage costs, privacy concerns.

**Open Source (Hugging Face, Llama)?** #flashcard
Requires GPU, full access to weights and internals, privacy, customization.

**Using a generative model for classification when a BERT-class model would be 10× cheaper and faster.?** #flashcard
Using a generative model for classification when a BERT-class model would be 10× cheaper and faster.

**Dismissing a model as "not an LLM" because it is small?** #flashcard
the relevant question is whether it effectively captures language for the target task.

**Relying on proprietary APIs when data cannot leave the organization (healthcare, legal, finance).?** #flashcard
Relying on proprietary APIs when data cannot leave the organization (healthcare, legal, finance).

**Subword tokenization (BERT)?** #flashcard
"CAPITALIZATION" → ['CA', '##PI', '##TA', '##L', '##I', '##Z', '##AT', '##ION']. Rare or compound words are decomposed into known subword units.

**Special tokens?** #flashcard
[CLS] (start of sequence), [SEP] (sequence separator), [PAD] (batch padding), [UNK] (out-of-vocabulary fallback).

**Vocabulary sizes?** #flashcard
typically 30,000–100,000 tokens.

**Capitalization matters?** #flashcard
the same word in different cases may produce different token IDs.

**Bag-of-Words for any task requiring semantic understanding?** #flashcard
it is a frequency table, not a meaning representation.

**Ignoring tokenization when debugging?** #flashcard
if a model performs badly on rare technical terms, inspect how those terms are tokenized first.

**Using static Word2Vec embeddings for tasks where word meaning shifts with context (sentiment, ambiguity resolution).?** #flashcard
Using static Word2Vec embeddings for tasks where word meaning shifts with context (sentiment, ambiguity resolution).

**Q (Query)?** #flashcard
what this token is looking for.

**K (Key)?** #flashcard
what each other token offers.

**V (Value)?** #flashcard
the actual content each token contributes.

**Division by √d_k?** #flashcard
as d_k grows, dot products grow in magnitude, pushing softmax into near-zero-gradient regions. Scaling by √d_k keeps variance around 1.

**Long sequences?** #flashcard
attention is the most computationally expensive part of the forward pass. Processing a 100,000-token document with full attention is prohibitively expensive.

**KV cache exhaustion?** #flashcard
for long conversations or large batch sizes, the KV cache fills GPU VRAM. This is why real-time chatbots have context limits in production.

**Abrupt output cutoff: when max_new_tokens is hit, generation stops mid-sentence. The model is not confused?** #flashcard
it hit the limit you set.

**Using GPT-4 to classify 10 million emails?** #flashcard
the cost is unjustifiable when a 110M-parameter BERT model achieves similar accuracy at 1/1000th the cost.

**Failing to constrain generative model output?** #flashcard
a generative model prompted to classify will sometimes produce a sentence of explanation instead of the label. Enforce strict output format.

**Zero-shot embedding classification on domain-specific jargon?** #flashcard
the embedding space may not meaningfully separate technical categories that a general model has not seen.

**K-Means requires you to specify the number of clusters in advance.?** #flashcard
K-Means requires you to specify the number of clusters in advance.

**HDBSCAN finds arbitrarily shaped clusters based on density.?** #flashcard
HDBSCAN finds arbitrarily shaped clusters based on density.

**Critically?** #flashcard
HDBSCAN assigns label -1 to points that don't belong to any dense cluster. These noise points are excluded from topic descriptions rather than being forced into the nearest cluster.

**Embedding quality ceiling?** #flashcard
if the underlying model does not understand your domain's jargon (legal, biomedical), semantically related documents will not embed near each other. Domain-specific embedding models are required.

**UMAP stochasticity?** #flashcard
UMAP results vary across runs. Cluster assignments can change between executions with no change in input data. Set a random seed for reproducibility.

**Trusting keyword lists blindly?** #flashcard
inspect actual documents within each cluster. A keyword like "patient" could describe a medical cluster or, coincidentally, a cluster about customer service.

**Embed query and all documents independently using a Bi-Encoder.?** #flashcard
Embed query and all documents independently using a Bi-Encoder.

**At query time?** #flashcard
embed the query, find the K nearest document embeddings by cosine similarity.

**Documents are pre-embedded and pre-indexed?** #flashcard
query time is just a vector lookup.

**Scales to millions of documents with ANN (Approximate Nearest Neighbor) indexes?** #flashcard
FAISS (Facebook AI Similarity Search), HNSW.

**Weakness?** #flashcard
query and answer may not be near each other in embedding space because they look different ("Who is the CEO?" ≠ "Satya Nadella").

**Cross-Encoder?** #flashcard
takes (query, document) as a pair, processes them together, outputs a relevance score.

**Far more accurate than Bi-Encoder because it can model the relationship between query and document.?** #flashcard
Far more accurate than Bi-Encoder because it can model the relationship between query and document.

**Cannot scale to millions of documents?** #flashcard
must process each pair at query time.

**Used as a second pass?** #flashcard
Dense Retrieval returns top-100 candidates; Reranker scores each and returns top-10.

**Bi-Encoder recall ceiling?** #flashcard
queries phrased unusually may not embed near the correct answer. Reranking is not optional for high-stakes retrieval.

**Chunk boundary artifacts?** #flashcard
if the answer spans two non-adjacent chunks, the model sees half the context from each and may miss the connection.

**Hallucination despite RAG?** #flashcard
if all retrieved chunks are irrelevant, the model may still generate a confident wrong answer. Enforce "If the provided text does not contain the answer, say so."

**Dense retrieval on exact-match queries?** #flashcard
semantic embeddings of "order #83291" and "order #83292" are essentially identical. Keyword search is required for exact matches.

**Pros?** #flashcard
maximum fidelity, nothing forgotten.

**Cons?** #flashcard
context window fills fast; cost and latency grow linearly with conversation length.

**Pros?** #flashcard
bounded context size, saves tokens.

**Cons?** #flashcard
summary call adds latency; summarization loses nuance; errors in the summary compound over time.

**Single-pass complex generation?** #flashcard
long-form coherent content requires sequential planning steps. One-shot prompts drift and lose consistency.

**Memory without pruning?** #flashcard
ConversationBufferMemory for long sessions will eventually hit the context window limit and crash the application.

**Summary compounding errors?** #flashcard
if the summary in turn 20 incorrectly summarizes turn 10, every subsequent turn inherits that error. Summaries must be validated or kept append-only.

**CoT without commitment?** #flashcard
some models generate reasoning steps and then ignore them when formulating the final answer. Force the model to reference its own chain explicitly in the final answer.

**Documents are embedded once, offline, and stored in a FAISS index.?** #flashcard
Documents are embedded once, offline, and stored in a FAISS index.

**At query time?** #flashcard
single embedding computation + nearest-neighbor lookup. Millisecond latency at millions-of-document scale.

**FAISS supports both exact search (IVF) and approximate search (HNSW). Approximate is faster; exact is more accurate.?** #flashcard
FAISS supports both exact search (IVF) and approximate search (HNSW). Approximate is faster; exact is more accurate.

**Input?** #flashcard
[CLS] query [SEP] document [SEP]

**Output?** #flashcard
a single relevance score.

**The model sees query-document interaction explicitly, not just their independent embeddings.?** #flashcard
The model sees query-document interaction explicitly, not just their independent embeddings.

**"Vastly improved results" compared to Bi-Encoder alone, at the cost of O(candidates) inference calls per query.?** #flashcard
"Vastly improved results" compared to Bi-Encoder alone, at the cost of O(candidates) inference calls per query.

**Generic Bi-Encoders on domain-specific corpora?** #flashcard
a general embedding model has no notion of your internal taxonomy. Fine-tuning the retrieval model on domain-specific Q-A pairs is necessary for high retrieval recall.

**Indexing without chunking?** #flashcard
embedding an entire 50-page document as a single vector produces a vector that averages over all topics in the document. The query "what is the refund policy?" will not find the correct paragraph if it is averaged with 49 other pages.

**Returning results past the relevance threshold?** #flashcard
for uncommon queries with no good match in the corpus, the nearest result may still be wrong. A distance cutoff is mandatory.

**Trained on millions of (image, caption) pairs.?** #flashcard
Trained on millions of (image, caption) pairs.

**Loss?** #flashcard
maximize similarity between matched image-text pairs, minimize similarity between mismatched pairs.

**Result: image embeddings and text embeddings are aligned?** #flashcard
embed("a photo of a dog") is near the embedding of an actual dog photo.

**Application?** #flashcard
zero-shot image classification. Classify by finding which text label embedding is nearest to the image embedding. No image-specific training required.

**Context window consumption?** #flashcard
image patches are tokens. A 224×224 image at 16×16 patches = 196 tokens, filling roughly 2-3% of a 8k context window per image. High-resolution images consume context rapidly.

**Resolution sensitivity?** #flashcard
ViT models expect specific input sizes (224×224 is standard). Images that are not resized to this specification before input produce degraded results or errors.

**CLIP on domain-specific imagery?** #flashcard
CLIP was trained on internet images. Medical imaging, satellite imagery, or industrial inspection images may not embed correctly without domain-specific fine-tuning.

**Layout vs. content?** #flashcard
multimodal models can read text within images, but spatial understanding of document layout (tables, forms, flowcharts) requires architectures specifically designed for document understanding, not just general ViT + LLM.

**Positive pairs?** #flashcard
(document A, document B) where A and B are semantically equivalent under your definition of similarity.

**Negative pairs?** #flashcard
(document A, document C) where A and C are not similar.

**Hard negatives?** #flashcard
(document A, document D) where D looks similar (shared keywords) but is actually wrong. Hard negatives are far more informative than random negatives.

**Before?** #flashcard
"I love this movie" ≈ "I hate this movie" (both are movie reviews)

**After?** #flashcard
"I love this movie" ≈ "This was wonderful" and far from "I hate this movie"

**Catastrophic forgetting?** #flashcard
aggressive fine-tuning can cause the model to lose general language understanding while becoming expert at the target task. Use a low learning rate and mix in general-domain examples.

**Low-quality negatives?** #flashcard
random negatives are easy to separate. The model learns trivially and fails on hard cases. Curate hard negatives explicitly.

**Misdefining similarity?** #flashcard
if you want to cluster by author style but your training pairs reflect topic similarity, the fine-tuned model will cluster by topic. The model learns what you show it, not what you intend.

**Evaluation against the wrong metric?** #flashcard
if you fine-tune for sentiment similarity but evaluate with a topic-similarity benchmark, the model looks like it got worse. Define your evaluation metric before defining your training pairs.

**Frozen BERT + Logistic Regression (Chapter 4)?** #flashcard
F1 = 0.80

**Full fine-tuning (this chapter)?** #flashcard
F1 = 0.85

**Freeze bottom N layers (basic grammar features?** #flashcard
these generalize well), fine-tune top layers (complex semantics — these are task-specific).

**Reduces compute while preserving most accuracy gain.?** #flashcard
Reduces compute while preserving most accuracy gain.

**Overfitting on small datasets?** #flashcard
fine-tuning a 110M-parameter model on 500 examples will memorize the training data. Monitor validation loss; stop training when it starts increasing.

**Catastrophic forgetting?** #flashcard
a model fine-tuned too aggressively on task-specific data loses general language understanding. Regularization (low learning rate, dropout) mitigates this.

**Deployment proliferation?** #flashcard
full fine-tuning produces a separate 12GB model file per task. Storing a fine-tuned BERT for 20 classification tasks = 240GB. Parameter-efficient fine-tuning (LoRA) addresses this but is not covered in depth in this chapter.

**Base Model?** #flashcard
trained on raw text, predicts next token. Useful for researchers, not users.

**Instruct Model?** #flashcard
fine-tuned to follow instructions. The product users interact with.

**Freeze all original model weights.?** #flashcard
Freeze all original model weights.

**For each target weight matrix W (usually Q and V in attention layers), add two small matrices?** #flashcard
W = W_0 + BA, where B is (d × r) and A is (r × d), with r << d.

**Only B and A are trained. If r = 8 and d = 4096, you train 2 × 4096 × 8 = 65,536 parameters instead of 4096 × 4096 = 16,777,216.?** #flashcard
Only B and A are trained. If r = 8 and d = 4096, you train 2 × 4096 × 8 = 65,536 parameters instead of 4096 × 4096 = 16,777,216.

**Reduces trainable parameters by 99%+ while achieving performance close to full fine-tuning.?** #flashcard
Reduces trainable parameters by 99%+ while achieving performance close to full fine-tuning.

**Quantize the frozen base model to 4-bit precision (reduces model size by 4×).?** #flashcard
Quantize the frozen base model to 4-bit precision (reduces model size by 4×).

**Apply LoRA adapters in 16-bit precision.?** #flashcard
Apply LoRA adapters in 16-bit precision.

**Fine-tune a 70B model on a single 48GB GPU.?** #flashcard
Fine-tune a 70B model on a single 48GB GPU.

**Input?** #flashcard
pairs of (chosen response, rejected response) for the same prompt.

**Loss?** #flashcard
directly optimize the model to prefer "chosen" over "rejected."

**No separate Reward Model. No PPO. One stable training loop.?** #flashcard
No separate Reward Model. No PPO. One stable training loop.

**Result?** #flashcard
equivalent alignment quality, significantly simpler implementation.

**SFT on low-quality data: the model learns to follow the format of the training data, including its errors and biases. Garbage in, garbage out?** #flashcard
at massive scale.

**LoRA rank selection: too small a rank (r=1, r=2) underfits?** #flashcard
the adapter cannot represent the required adaptations. Too large a rank approaches full fine-tuning. r=8 or r=16 are common starting points.

**DPO over-optimization (Alignment Tax)?** #flashcard
heavily aligning the model to human preference data can reduce output diversity. The model converges to "safe and agreeable" responses and loses capability on tasks requiring creative or unconventional reasoning. Heavy alignment can measurably degrade coding and reasoning benchmark scores.

**Base model as the wrong starting point?** #flashcard
fine-tuning a base model on instruction data without first applying SFT from the provider's pipeline produces erratic behavior. Always start SFT from the base model, not from a partially-aligned checkpoint you cannot inspect.
