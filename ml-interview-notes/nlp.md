# NLP

---

# Q1: What are the advantages of Transformers over traditional sequence-to-sequence models?

## 1. 🔹 Direct Answer
**Transformers** replace recurrence with **self-attention**—**parallelize** over sequence length, **capture long-range** dependencies in **O(1)** layers (vs RNN **O(T)** sequential steps), and scale with **data/GPU**. **Seq2seq** Transformers (encoder-decoder) avoid **bottleneck** of fixed context in LSTM.

## 2. 🔹 Intuition
RNNs read left-to-right slowly; attention lets every token **look at** every other token directly.

## 3. 🔹 Deep Dive
**Multi-head** attention learns multiple relation types; **positional** encodings inject order.

## 4. 🔹 Practical Perspective
**Pretraining** + **fine-tuning** dominates NLP; RNNs still used for **tiny** edge or streaming with constraints.

## 5. 🔹 Code Snippet
```text
Attention(Q,K,V) = softmax(QK^T/sqrt(d)) V
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Cost? **A:** O(n²) attention memory—long context expensive.

## 7. 🔹 Common Mistakes
Claiming RNNs never used—still valid for **low-latency** streaming with constraints.

## 8. 🔹 Comparison / Connections
CNNs for local patterns, State Space Models (Mamba) for long seq.

## 9. 🔹 One-line Revision
Transformers parallelize training and model long-range deps via attention—dominant with scale.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q2: What are the limitations of Transformers, and how can they be addressed?

## 1. 🔹 Direct Answer
**Quadratic** attention cost in sequence length, **data/compute** hungry, **position** extrapolation limits, **hallucination** in generative use. Mitigations: **sparse/local/windowed** attention, **FlashAttention**, **long-context** RoPE scaling, **RAG**, **KV cache** optimizations, **distillation** for efficiency.

## 2. 🔹 Intuition
Full attention is all-to-all—doesn’t scale to million tokens naively.

## 3. 🔹 Deep Dive
**Inductive bias** weaker than CNNs for vision—needs **more data** or **hybrids**.

## 4. 🔹 Practical Perspective
**Latency** in autoregressive decoding—**speculative decoding**, **batching**.

## 5. 🔹 Code Snippet
```text
complexity: O(L^2 * d) naive attention memory
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Linear attention? **A:** Approximate softmax-kernel methods—trade accuracy.

## 7. 🔹 Common Mistakes
Ignoring **inference** cost when praising Transformers.

## 8. 🔹 Comparison / Connections
RNN memory O(L), SSMs.

## 9. 🔹 One-line Revision
Transformers trade flexibility for quadratic cost and data hunger—mitigate with efficient attention and systems tricks.

## 10. 🔹 Difficulty Tag
🟣 Hard

---

# Q3: What is BERT, and how does it improve language understanding?

## 1. 🔹 Direct Answer
**BERT** is **bidirectional Transformer encoder** pretrained with **MLM** (mask prediction) + **NSP** (sentence pair). **Fine-tuning** on downstream tasks yields strong **contextual** representations vs static embeddings.

## 2. 🔹 Intuition
Reads **full context** both directions—better word sense than left-only LSTM.

## 3. 🔹 Deep Dive
**[CLS]** token for classification; **token-level** outputs for NER.

## 4. 🔹 Practical Perspective
Superseded by **RoBERTa** (better training), **DeBERTa**, etc.—ideas similar.

## 5. 🔹 Code Snippet
```python
from transformers import AutoModel
model = AutoModel.from_pretrained("bert-base-uncased")
```

## 6. 🔹 Interview Follow-ups
1. **Q:** vs GPT? **A:** BERT encoder bidirectional; GPT decoder causal for generation.

## 7. 🔹 Common Mistakes
Using BERT for **open-ended generation** without autoregressive decoder.

## 8. 🔹 Comparison / Connections
T5, ELECTRA pretraining.

## 9. 🔹 One-line Revision
BERT is bidirectional masked LM pretraining for rich contextual embeddings—fine-tune for NLU tasks.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q4: How are Transformers trained (pre-training and fine-tuning)?

## 1. 🔹 Direct Answer
**Pre-training**: self-supervised on large corpus (**MLM**, **causal LM**, **span corruption**) to learn **general** representations. **Fine-tuning**: supervised data for **task heads** (classification, QA) with smaller LR; or **instruction tuning** / **RLHF** for alignment.

## 2. 🔹 Intuition
Pretrain on **cheap** text; specialize on **expensive** labels.

## 3. 🔹 Deep Dive
**Transfer** reduces sample complexity on downstream; **catastrophic forgetting** mitigated by **adapters**, **LoRA**.

## 4. 🔹 Practical Perspective
**Compute** budget drives model size; **data quality** matters as much as size.

## 5. 🔹 Code Snippet
```text
pretrain: minimize LM loss on web text ; finetune: minimize task loss on labeled data
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Continued pretraining? **A:** Domain adaptation before task fine-tune.

## 7. 🔹 Common Mistakes
Fine-tuning with **too high** LR destroying pretrained features.

## 8. 🔹 Comparison / Connections
Few-shot prompting vs fine-tuning.

## 9. 🔹 One-line Revision
Pretrain self-supervised on broad text; fine-tune or align on task-specific or human preference data.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q5: Explain transfer learning in the context of Transformers.

## 1. 🔹 Direct Answer
**Reuse** pretrained **weights** (language modeling) as **initialization** for downstream NLP—**faster convergence**, **better** accuracy with **less** labeled data than training from scratch.

## 2. 🔹 Intuition
General language structure is **shared** across tasks—only task head adapts.

## 3. 🔹 Deep Dive
**Feature extraction**: freeze backbone; **full fine-tune**: update all layers; **partial**: top layers only.

## 4. 🔹 Practical Perspective
**Domain mismatch** (legal vs tweets)—**continued pretraining** or **adapters**.

## 5. 🔹 Code Snippet
```python
for p in model.base.parameters(): p.requires_grad = False
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Prompting vs fine-tuning? **A:** Zero-shot fast; fine-tune for reliable specialized behavior.

## 7. 🔹 Common Mistakes
Fine-tuning on **tiny** data without regularization—overfits.

## 8. 🔹 Comparison / Connections
Computer vision ImageNet transfer, multitask learning.

## 9. 🔹 One-line Revision
Transformer transfer learning reuses pretrained language knowledge via heads, adapters, or LoRA.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q6: Describe the process of text generation using Transformer-based language models.

## 1. 🔹 Direct Answer
**Autoregressive** decoding: start with **prompt tokens**, repeatedly **predict next token** distribution from **causal** Transformer, **sample** (greedy, top-k, top-p, temperature), **append** token until **EOS** or max length. **KV-cache** stores past keys/values for efficiency.

## 2. 🔹 Intuition
Like **chain of predictions**—each word conditions on all previous.

## 3. 🔹 Deep Dive
**Training**: maximize log-likelihood of token sequence. **Inference**: **sampling** controls diversity vs quality.

## 4. 🔹 Practical Perspective
**Stop** criteria, **repetition** penalties, **safety** filters post-hoc.

## 5. 🔹 Code Snippet
```python
# Hugging Face generate()
outputs = model.generate(input_ids, max_new_tokens=100, do_sample=True, top_p=0.9, temperature=0.8)
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Beam search? **A:** Good for translation; open-ended text often worse than sampling.

## 7. 🔹 Common Mistakes
Ignoring **context length** limits and **truncation** side effects.

## 8. 🔹 Comparison / Connections
Diffusion for text (non-autoregressive research), constrained decoding.

## 9. 🔹 One-line Revision
LM generation is iterative next-token prediction with caching—sampling strategy shapes output style.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q7: What are Seq2Seq models?

## 1. 🔹 Direct Answer
**Seq2seq** maps **variable-length input** sequence to **variable-length output** (MT, summarization). Classically **encoder** (RNN/CNN/Transformer) builds representation, **decoder** generates output tokens **autoregressively**; **attention** links decoder to encoder states.

## 2. 🔹 Intuition
Compression of input meaning into **context** then **expansion** to output language.

## 3. 🔹 Deep Dive
**Teacher forcing** during training; **beam search** at inference.

## 4. 🔹 Practical Perspective
**T5/BART**: text-to-text unifies tasks with **prefix** prompts.

## 5. 🔹 Code Snippet
```text
P(y|x) = Π P(y_t | y_<t, Encoder(x))
```

## 6. 🔹 Interview Follow-ups
1. **Q:** vs encoder-only? **A:** Encoder-only for classification; seq2seq for generation.

## 7. 🔹 Common Mistakes
Confusing seq2seq with **encoder-only** BERT.

## 8. 🔹 Comparison / Connections
Pointer networks, Whisper (speech seq2seq).

## 9. 🔹 One-line Revision
Seq2seq is encoder-decoder for input sequence to output sequence—attention bridges both sides.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q8: Compare N-gram models and deep learning models (trade-offs).

## 1. 🔹 Direct Answer
**N-grams**: **sparse**, **count-based**, **local** context (n−1), **fast**, **interpretable**, **data-efficient** for tiny corpora, **poor** generalization. **Neural/DL**: **dense** embeddings, **long-range** patterns, **scalable** with data/compute, **black-box**, need **more** data and **hardware**.

## 2. 🔹 Intuition
N-grams memorize **exact** short fragments; neural models **generalize** similar words.

## 3. 🔹 Deep Dive
N-gram **backoff/Kneser-Ney** smoothing; neural **perplexity** lower with scale.

## 4. 🔹 Practical Perspective
N-grams still in **ASR** **language models** hybrid; **debugging** with n-gram sanity checks.

## 5. 🔹 Code Snippet
```text
P(w_i | w_{i-n+1}^{i-1}) from counts + smoothing
```

## 6. 🔹 Interview Follow-ups
1. **Q:** When n-grams win? **A:** Tiny data, latency-critical simple LM, interpretability.

## 7. 🔹 Common Mistakes
Saying n-grams useless—still baselines and components.

## 8. 🔹 Comparison / Connections
Katz smoothing, neural LM history.

## 9. 🔹 One-line Revision
N-grams are simple local count models; deep models learn distributed representations and scale—but need data.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q9: What is the n-gram model?

## 1. 🔹 Direct Answer
**n-gram model** estimates **P(w_i | w_{i-n+1}^{i-1})** using **relative frequencies** of n-token sequences—**Markov** assumption of order **n−1**. **Unigram, bigram, trigram** common.

## 2. 🔹 Intuition
Predict next word from **last few** words only—simple and fast.

## 3. 🔹 Deep Dive
**Smoothing** (add-α, Kneser-Ney) for unseen n-grams; **perplexity** evaluation.

## 4. 🔹 Practical Perspective
**Sparsity** explodes with n—**backoff** essential.

## 5. 🔹 Code Snippet
```python
from collections import Counter
bigrams = Counter(zip(words, words[1:]))
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Kneser-Ney why? **A:** Better backoff for **continuation** probability.

## 7. 🔹 Common Mistakes
Zero probabilities without smoothing.

## 8. 🔹 Comparison / Connections
Neural language models, skip-grams.

## 9. 🔹 One-line Revision
n-grams estimate next-token probs from short context counts—Markov + smoothing.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q10: What is TF-IDF, and how does it differ from word embeddings?

## 1. 🔹 Direct Answer
**TF-IDF** weights terms by **frequency in document** × **inverse rarity in corpus**—**sparse**, **high-dimensional**, **interpretable**, **no** learning. **Word embeddings** are **dense**, **learned** vectors capturing **similarity** from **context** (prediction-based or count-based).

## 2. 🔹 Intuition
TF-IDF highlights **distinctive** words for retrieval; embeddings capture **semantic** neighborhood.

## 3. 🔹 Deep Dive
TF-IDF: **tf_ij** × **log(N/df_j)**; embeddings: **cosine** similarity in **d**-dim.

## 4. 🔹 Practical Perspective
TF-IDF still strong **baseline** for **search**; embeddings for **semantic** retrieval and **downstream** NN.

## 5. 🔹 Code Snippet
```python
from sklearn.feature_extraction.text import TfidfVectorizer
vec = TfidfVectorizer(max_features=10000)
X = vec.fit_transform(corpus)
```

## 6. 🔹 Interview Follow-ups
1. **Q:** BM25? **A:** Probabilistic IR improvement over TF-IDF with length norm.

## 7. 🔹 Common Mistakes
Using TF-IDF for **synonym** generalization—limited.

## 8. 🔹 Comparison / Connections
BM25, dense passage retrieval (DPR).

## 9. 🔹 One-line Revision
TF-IDF is sparse lexical weighting; embeddings are dense learned semantics—different retrieval regimes.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q11: What is Bag-of-Words?

## 1. 🔹 Direct Answer
**BoW** represents text as ** multiset of tokens**—**counts** or **binary** presence, **discarding order**. Simple **baseline** for classification; loses **syntax** and **negation** unless **n-grams** added.

## 2. 🔹 Intuition
“Which words show up” without “how they relate.”

## 3. 🔹 Deep Dive
Vector length = **vocab size**; **sparse** matrices.

## 4. 🔹 Practical Perspective
Works surprisingly well for **topic** signal; combine with **char n-grams** for robustness.

## 5. 🔹 Code Snippet
```python
from sklearn.feature_extraction.text import CountVectorizer
CountVectorizer(binary=True)
```

## 6. 🔹 Interview Follow-ups
1. **Q:** vs TF-IDF? **A:** TF-IDF downweights common words.

## 7. 🔹 Common Mistakes
Ignoring **stop words** / **normalization** needs per task.

## 8. 🔹 Comparison / Connections
SetencePiece, hashing vectorizer.

## 9. 🔹 One-line Revision
BoW is orderless token frequency vector—fast baseline but blind to structure.

## 10. 🔹 Difficulty Tag
🟢 Easy

---

# Q12: What is perplexity used for in NLP?

## 1. 🔹 Direct Answer
**Perplexity** = **exp(cross-entropy loss)** = **2^H** bits if log2—measures how **surprised** a language model is by test data. **Lower** is better; **intrinsic** LM quality metric (not always downstream quality).

## 2. 🔹 Intuition
“If model is perplexed 50,” average **uncertainty** like uniform over ~50 choices.

## 3. 🔹 Deep Dive
For token-level CE: **PP** = exp(−(1/N) Σ log p(w_i|context)).

## 4. 🔹 Practical Perspective
Compare LMs **same tokenizer/vocab**; correlate with **BLEU** imperfectly.

## 5. 🔹 Code Snippet
```python
import math
pp = math.exp(total_nll / num_tokens)
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Bits per character? **A:** Related compression viewpoint.

## 7. 🔹 Common Mistakes
Comparing perplexity across **different** tokenizations.

## 8. 🔹 Comparison / Connections
Bits-per-byte, cross-entropy.

## 9. 🔹 One-line Revision
Perplexity exponentiates average token negative log-likelihood—standard LM comparison at same tokenization.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q13: What is stemming vs lemmatization?

## 1. 🔹 Direct Answer
**Stemming** heuristically **chops** suffixes (**running→run**)—fast, **error-prone**. **Lemmatization** uses **vocabulary/morphology** to canonical **lemma** (**better→good**)—slower, **more accurate**.

## 2. 🔹 Intuition
Stemming is **rough** cuts; lemmatization is **dictionary**-aware normalization.

## 3. 🔹 Deep Dive
Porter/Snowball stemmers; WordNet lemmatizer needs **POS** tags for accuracy.

## 4. 🔹 Practical Perspective
Modern **subword** tokenizers (BPE) reduce need; still used in **IR** classical pipelines.

## 5. 🔹 Code Snippet
```python
from nltk.stem import PorterStemmer, WordNetLemmatizer
```

## 6. 🔹 Interview Follow-ups
1. **Q:** When skip? **A:** Neural models often prefer **raw** text with BPE.

## 7. 🔹 Common Mistakes
Over-stemming destroying **distinct** meanings.

## 8. 🔹 Comparison / Connections
Tokenization, normalization.

## 9. 🔹 One-line Revision
Stemming is crude suffix stripping; lemmatization maps to dictionary lemmas—choose based on downstream.

## 10. 🔹 Difficulty Tag
🟢 Easy

---

# Q14: What is Latent Semantic Indexing (LSI)?

## 1. 🔹 Direct Answer
**LSI** applies **SVD** to **term-document matrix** to discover **latent topics**—**reduces dimension**, captures **synonymy** by **co-occurrence**. Predecessor to modern **embeddings** for IR.

## 2. 🔹 Intuition
Projects bag-of-words into **smaller** space where related terms **cluster**.

## 3. 🔹 Deep Dive
**Truncated SVD** = PCA for sparse matrices; **orthogonal** factors.

## 4. 🔹 Practical Perspective
Superseded by **neural** embeddings + **ANN** search for semantic retrieval.

## 5. 🔹 Code Snippet
```python
from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=100)
X_latent = svd.fit_transform(tfidf_matrix)
```

## 6. 🔹 Interview Follow-ups
1. **Q:** vs LDA? **A:** LDA generative topic model with Dirichlet priors—different assumptions.

## 7. 🔹 Common Mistakes
Confusing LSI with **LSA** naming—often used interchangeably.

## 8. 🔹 Comparison / Connections
PCA, topic models, dense retrieval.

## 9. 🔹 One-line Revision
LSI is SVD on term-document matrix for latent topical structure—historical semantic IR technique.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q15: What is dependency parsing?

## 1. 🔹 Direct Answer
**Dependency parsing** finds **directed syntactic** relations (subject, object, modifier) between **words**—**tree** over tokens. Used for **information extraction**, **QA**, **opinion** mining.

## 2. 🔹 Intuition
Who did what to whom—**structure** beyond bag of words.

## 3. 🔹 Deep Dive
**Transition-based** or **graph-based** parsers; **UD** universal dependencies standard.

## 4. 🔹 Practical Perspective
**Neural** parsers (biaffine) SOTA; **multilingual** models exist.

## 5. 🔹 Code Snippet
```python
import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp("The cat sat on the mat.")
for t in doc: print(t.text, t.dep_, t.head.text)
```

## 6. 🔹 Interview Follow-ups
1. **Q:** vs constituency? **A:** Phrase-structure trees vs word-to-word deps.

## 7. 🔹 Common Mistakes
Confusing **semantic** roles with **syntactic** deps.

## 8. 🔹 Comparison / Connections
Semantic parsing, SRL.

## 9. 🔹 One-line Revision
Dependency parsing extracts directed grammatical relations between words—key for structured NLP.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q16: What are some approaches for text summarization?

## 1. 🔹 Direct Answer
**Extractive**: select **sentences** (TextRank, supervised scoring). **Abstractive**: **generate** summary (seq2seq, Transformers). **Hybrid**: extract then rewrite. **Eval**: **ROUGE** (n-gram overlap), **BERTScore**, human judgments.

## 2. 🔹 Intuition
Extractive safer (faithful); abstractive **fluent** but **hallucination** risk.

## 3. 🔹 Deep Dive
**Pointer-generator** networks copy from source; **RL** with ROUGE reward—unstable.

## 4. 🔹 Practical Perspective
**News** vs **meetings**—domain matters; **length** constraints.

## 5. 🔹 Code Snippet
```text
summarization: P(summary|doc) with seq2seq + attention + copy
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Factuality? **A:** RAG-style grounding, entailment models.

## 7. 🔹 Common Mistakes
ROUGE-only optimization without **factuality** checks.

## 8. 🔹 Comparison / Connections
LLM summarization, long-context models.

## 9. 🔹 One-line Revision
Extractive for fidelity; abstractive for fluency—evaluate with ROUGE + faithfulness checks.

## 10. 🔹 Difficulty Tag
🟣 Hard

---

# Q17: What are word embeddings?

## 1. 🔹 Direct Answer
**Word embeddings** map tokens to **dense** **ℝ^d** vectors so **similar** words (distributionally) are **close** in space—learned via **Word2Vec**, **GloVe**, **fastText**, or **contextual** ELMo/BERT embeddings.

## 2. 🔹 Intuition
One-hot is huge and orthogonal; embeddings **share** statistical strength.

## 3. 🔹 Deep Dive
**Static** (Word2Vec) vs **contextual** (ELMo/BERT)—polysemy handling.

## 4. 🔹 Practical Perspective
**Subword** fastText helps **OOV**; **normalize** vectors for cosine similarity.

## 5. 🔹 Code Snippet
```python
import gensim.downloader as api
w2v = api.load("word2vec-google-news-300")
w2v.most_similar("king")
```

## 6. 🔹 Interview Follow-ups
1. **Q:** king - man + woman? **A:** Linear analogies—illustrates structure.

## 7. 🔹 Common Mistakes
Biases in embeddings reflecting corpus—**fairness** risk.

## 8. 🔹 Comparison / Connections
One-hot, TF-IDF, sentence embeddings.

## 9. 🔹 One-line Revision
Word embeddings are dense vectors capturing distributional similarity—static or contextual.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q18: What is Word2Vec?

## 1. 🔹 Direct Answer
**Word2Vec** learns embeddings by **predicting** context (**skip-gram**) or word from context (**CBOW**) with **negative sampling**—**shallow** neural model, **fast**, **static** embeddings per word type.

## 2. 🔹 Intuition
Words in **similar contexts** get **similar vectors**.

## 3. 🔹 Deep Dive
**Negative sampling** approximates softmax; **subsampling** frequent words.

## 4. 🔹 Practical Perspective
Still used as **init** or **small-data** tool; **subword** fastText often better for morphology.

## 5. 🔹 Code Snippet
```python
from gensim.models import Word2Vec
model = Word2Vec(sentences, vector_size=100, window=5, min_count=2, sg=1)
```

## 6. 🔹 Interview Follow-ups
1. **Q:** CBOW vs skip-gram? **A:** Skip-gram better for rare words (more context prediction).

## 7. 🔹 Common Mistakes
Expecting **sense** disambiguation—single vector per word.

## 8. 🔹 Comparison / Connections
GloVe (global count matrix), ELMo.

## 9. 🔹 One-line Revision
Word2Vec is skip-gram/CBOW with negative sampling learning static word vectors from local context.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q19: What is t-SNE, and how is it used for NLP?

## 1. 🔹 Direct Answer
**t-SNE** nonlinearly embeds high-dim points to **2D/3D** preserving **local** neighborhoods—**visualize** word/document **clusters**. **Stochastic**, **nonparametric**—**not** for **rigorous** distance or **downstream** features.

## 2. 🔹 Intuition
Pull similar points close, dissimilar apart—**perplexity** controls neighborhood size.

## 3. 🔹 Deep Dive
**KL divergence** between high-dim Gaussian similarities and low-dim Student-t similarities.

## 4. 🔹 Practical Perspective
Use for **plots** only; **UMAP** often preferred for **global** structure trade-offs.

## 5. 🔹 Code Snippet
```python
from sklearn.manifold import TSNE
Z = TSNE(n_components=2, perplexity=30).fit_transform(X)
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Perplexity? **A:** Balance local vs global—try multiple values.

## 7. 🔹 Common Mistakes
Interpreting **inter-cluster distances** as meaningful.

## 8. 🔹 Comparison / Connections
PCA (linear), UMAP.

## 9. 🔹 One-line Revision
t-SNE visualizes high-dim word vectors in 2D for exploration—don’t use coordinates as features.

## 10. 🔹 Difficulty Tag
🟡 Medium

---
