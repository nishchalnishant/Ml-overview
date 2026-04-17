# NLP

Good NLP interview answers should make the progression clear: sparse statistical methods, learned embeddings, sequence models, and then transformers.

---

# Q1: What are the advantages of Transformers over traditional sequence-to-sequence models?

**Interview-ready answer**

Transformers handle long-range dependencies better than traditional seq2seq models based on RNNs because attention lets every token interact directly with every other token. They are also highly parallelizable during training since they do not process tokens strictly one at a time. That is why they scale much better with data and compute and became the dominant architecture for modern NLP.

---

# Q2: What are the limitations of Transformers, and how can they be addressed?

**Interview-ready answer**

The main limitation is the cost of attention, which grows roughly quadratically with sequence length. They also need a lot of data and compute, and their large context windows can still be imperfect in practice despite the architecture's flexibility. Common mitigations include sparse or linear attention variants, chunking, retrieval augmentation, distillation, quantization, and domain-specific pretraining or fine-tuning.

---

# Q3: What is BERT, and how does it improve language understanding?

**Interview-ready answer**

BERT is an encoder-only transformer pretrained to understand text bidirectionally. It is trained with objectives like masked language modeling so it learns contextual representations that depend on both left and right context. That was a major improvement over earlier embeddings because the representation of a word like "bank" can change depending on the sentence, which significantly improves language understanding tasks such as classification, question answering, and retrieval.

---

# Q4: How are Transformers trained (pre-training and fine-tuning)?

**Interview-ready answer**

Transformers are usually pretrained on large unlabeled corpora using self-supervised objectives such as next-token prediction or masked token prediction. That phase teaches broad language structure and semantic relationships. They are then fine-tuned or adapted on a downstream task with supervised data, instruction data, or task-specific objectives. The big advantage is that pretraining transfers useful representations so downstream tasks need far less labeled data than training from scratch.

---

# Q5: Explain transfer learning in the context of Transformers.

**Interview-ready answer**

Transfer learning means reusing a pretrained transformer and adapting it to a new task rather than learning all parameters from scratch. In NLP this is powerful because pretraining already captures syntax, semantics, and some world knowledge. The adaptation can be full fine-tuning, parameter-efficient tuning, or even prompt-based use depending on the model and task.

---

# Q6: Describe the process of text generation using Transformer-based language models.

**Interview-ready answer**

Text generation with a decoder-style transformer is autoregressive. The model takes the current token sequence, predicts a probability distribution for the next token, selects or samples one token, appends it, and repeats. The quality of generation depends not only on the model but also on decoding strategy such as greedy decoding, beam search, temperature, top-k, or nucleus sampling.

---

# Q7: What are Seq2Seq models?

**Interview-ready answer**

Seq2Seq models map one sequence to another, such as translating a sentence in one language into another language. The classic version uses an encoder to summarize the input and a decoder to generate the output. Modern transformer encoder-decoder models are still seq2seq models conceptually; they just use attention instead of recurrence as the core mechanism.

---

# Q8: Compare N-gram models and deep learning models (trade-offs).

**Interview-ready answer**

N-gram models are simple probabilistic models that predict the next word from a short local context. They are fast, interpretable, and work surprisingly well as baselines on constrained problems, but they suffer badly from sparsity and cannot capture long-range context or meaning. Deep learning models learn dense representations and broader contextual structure, which makes them much more powerful, but they require far more data, compute, and engineering effort.

---

# Q9: What is the n-gram model?

**Interview-ready answer**

An n-gram model estimates the probability of a token based on the previous `n - 1` tokens. For example, a trigram model predicts the next word from the previous two words. Its main limitation is that it only sees a short fixed window and struggles with unseen combinations, which is why smoothing and backoff are important in classical language modeling.

---

# Q10: What is TF-IDF, and how does it differ from word embeddings?

**Interview-ready answer**

TF-IDF represents a document by weighting terms according to how frequent they are in the document and how rare they are across the corpus. It is sparse, interpretable, and works well for many search and classification baselines. Word embeddings are dense learned vectors that capture semantic similarity between words. So TF-IDF is count-based and document-centric, while embeddings are learned, dense, and better at representing semantic relationships.

---

# Q11: What is Bag-of-Words?

**Interview-ready answer**

Bag-of-Words represents a document by the counts or presence of vocabulary terms, ignoring word order. It is simple and often surprisingly strong as a baseline, especially with linear models, but it loses syntax, phrase structure, and context. The phrase "dog bites man" and "man bites dog" look identical under pure Bag-of-Words.

---

# Q12: What is perplexity used for in NLP?

**Interview-ready answer**

Perplexity measures how well a language model predicts a sequence. Lower perplexity means the model assigns higher probability to the observed text and is, in that sense, less "surprised." It is useful for comparing language models under the same tokenization and evaluation setup, but it does not always correlate perfectly with downstream quality, especially for instruction following or human preference.

---

# Q13: What is stemming vs lemmatization?

**Interview-ready answer**

Stemming reduces words to crude root forms by chopping endings, while lemmatization maps words to proper dictionary base forms using linguistic rules and often context. Stemming is faster and rougher; lemmatization is cleaner and more semantically faithful. In modern deep NLP this preprocessing is often less central, but it still matters for search, classical pipelines, and low-resource settings.

---

# Q14: What is Latent Semantic Indexing (LSI)?

**Interview-ready answer**

LSI is a dimensionality-reduction approach for text that applies matrix factorization, typically truncated SVD, to a term-document matrix. The goal is to uncover latent semantic structure and reduce sparsity so related words and documents can be represented closer together even if they do not share identical surface forms. It was an important precursor to modern embedding methods.

---

# Q15: What is dependency parsing?

**Interview-ready answer**

Dependency parsing analyzes the grammatical structure of a sentence by identifying head-dependent relationships between words, such as subject, object, and modifier links. It is useful when tasks depend heavily on syntax, relation extraction, or structured understanding. Even though transformers reduced reliance on explicit pipelines, dependency parsing remains important in linguistically grounded NLP tasks.

---

# Q16: What are some approaches for text summarization?

**Interview-ready answer**

Text summarization can be extractive or abstractive. Extractive methods select important sentences or spans from the original text, which is simpler and usually safer. Abstractive methods generate a new summary in natural language, which is more flexible but also more prone to factual errors. In practice, modern summarization often uses pretrained seq2seq transformers with task-specific prompting or fine-tuning.

---

# Q17: What are word embeddings?

**Interview-ready answer**

Word embeddings are dense vector representations of words learned so that words appearing in similar contexts get similar vectors. They solved a major limitation of one-hot representations by introducing similarity and geometry into the representation space. Static embeddings like Word2Vec and GloVe assign one vector per word type, while contextual models like BERT produce different representations for the same token depending on the sentence.

---

# Q18: What is Word2Vec?

**Interview-ready answer**

Word2Vec is a family of shallow neural models that learn word embeddings from context. The two classic variants are CBOW, which predicts a word from its context, and Skip-gram, which predicts surrounding words from a center word. Its importance is historical and practical: it showed that simple self-supervised objectives can produce semantically meaningful vector spaces.

---

# Q19: What is t-SNE, and how is it used for NLP?

**Interview-ready answer**

t-SNE is a non-linear dimensionality reduction technique mainly used for visualization. In NLP, it is often used to plot embeddings or document representations in two dimensions to inspect clustering or neighborhood structure qualitatively. The important caveat is that t-SNE is for visualization, not for downstream modeling, and the plot can change a lot with hyperparameters like perplexity and random seed.
