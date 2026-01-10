# NLP & Transformers: 40+ Questions

---

##  Text Preprocessing

**1. What is Tokenization?**
> Splitting text into tokens (words, subwords, or characters).

**2. What is BPE (Byte Pair Encoding)?**
> Iteratively merges frequent character pairs. Creates subword vocabulary. Handles OOV words.

**3. What is WordPiece?**
> Similar to BPE but uses likelihood-based merging. Used by BERT.

**4. What is SentencePiece?**
> Language-agnostic tokenizer. Treats text as raw bytes. No need for pre-tokenization.

**5. Why use subword tokenization over word-level?**
> Handles OOV words by breaking into known subwords. Smaller vocabulary size.

**6. What preprocessing steps are common for NLP?**
> Lowercasing, removing punctuation/URLs, handling contractions, tokenization.

**7. What is Stemming vs Lemmatization?**
> **Stemming**: Chops word endings (faster, cruder). **Lemmatization**: Reduces to dictionary form (slower, accurate).

**8. Should you lowercase text for BERT?**
> Depends. BERT-base-uncased: yes. BERT-base-cased: no. Use cased for NER.

---

##  Word Representations

**9. What is TF-IDF?**
> Term Frequency × Inverse Document Frequency. Weighs words by importance in corpus.

**10. What is Word2Vec?**
> Neural network that learns word embeddings. Two methods: Skip-gram, CBOW.

**11. What is Skip-gram vs CBOW?**
> **Skip-gram**: Predict context from word. **CBOW**: Predict word from context.

**12. What is GloVe?**
> Learns embeddings from co-occurrence matrix. Captures global statistics.

**13. What are Contextual Embeddings?**
> Same word gets different vectors based on context (ELMo, BERT). Solves polysemy.

**14. Why are BERT embeddings better than Word2Vec?**
> Context-dependent. "Bank" in "river bank" vs "money bank" gets different vectors.

**15. What is the embedding dimension typically?**
> 300 for Word2Vec/GloVe. 768 for BERT-base. 1024 for BERT-large.

---

##  Transformer Architecture

**16. What is Self-Attention?**
> Each token attends to all tokens. Computes relevance-weighted sum.

**17. What is the Attention formula?**
> $Attention(Q,K,V) = Softmax(\frac{QK^T}{\sqrt{d_k}})V$

**18. Why scale by $\sqrt{d_k}$?**
> Prevents large dot products from saturating softmax, which kills gradients.

**19. What is Multi-Head Attention?**
> Multiple parallel attention operations with different projections. Captures varied relationships.

**20. What is the number of heads typically?**
> 12 for BERT-base, 16 for BERT-large.

**21. What is Positional Encoding?**
> Injects position information. Sinusoidal (fixed) or learned.

**22. Why is positional encoding needed?**
> Attention is permutation-invariant. Doesn't know token order without position info.

**23. What is Masked (Causal) Attention?**
> Prevents attending to future tokens. Used in GPT for autoregressive generation.

**24. What is Cross-Attention?**
> Query from decoder, Key/Value from encoder. Connects encoder-decoder.

---

##  BERT Deep Dive

**25. What is BERT?**
> Bidirectional Encoder Representations from Transformers. Encoder-only.

**26. What are BERT's pre-training objectives?**
> **MLM**: Predict masked tokens. **NSP**: Predict if sentence B follows A.

**27. What is Masked Language Modeling (MLM)?**
> 15% of tokens are masked. Model predicts the original token.

**28. What is [CLS] token used for?**
> Classification token. Its final embedding represents the whole sequence.

**29. What is [SEP] token used for?**
> Separates two sentences in sentence-pair tasks.

**30. How do you fine-tune BERT for classification?**
> Add classification head on [CLS] token. Fine-tune all layers.

**31. What is Feature Extraction vs Fine-tuning?**
> **Feature Extraction**: Freeze BERT, train only classifier. **Fine-tuning**: Update all weights.

**32. What is RoBERTa?**
> BERT trained longer, on more data, without NSP. Dynamic masking.

---

##  GPT & Generation

**33. What is GPT?**
> Generative Pre-trained Transformer. Decoder-only, autoregressive.

**34. How is GPT trained?**
> Next-token prediction: Predict token t given tokens 1 to t-1.

**35. What is Causal Language Modeling?**
> Same as next-token prediction. Only sees past tokens.

**36. BERT vs GPT: when to use which?**
> **BERT**: Understanding (NER, sentiment). **GPT**: Generation (chatbot, summarization).

**37. What is Temperature in generation?**
> Controls randomness. Low temp = deterministic. High temp = diverse.

**38. What is Top-k sampling?**
> Sample from top k most likely tokens.

**39. What is Top-p (Nucleus) sampling?**
> Sample from smallest set of tokens whose cumulative probability ≥ p.

**40. What is Beam Search?**
> Keeps top k sequences at each step. More coherent than greedy, less diverse than sampling.

---

##  NLP Tasks

**41. What is Sentiment Analysis?**
> Classify text as positive/negative/neutral. Classification on [CLS] embedding.

**42. What is Named Entity Recognition (NER)?**
> Identify entities (Person, Location, Organization) in text. Token classification.

**43. What is Question Answering (Extractive)?**
> Predict start and end indices of answer span in context.

**44. What is Text Summarization?**
> Generate shorter version of document. Abstractive (generate new text) or Extractive (select sentences).

**45. What is Machine Translation?**
> Convert text from one language to another. Encoder-decoder architecture.

**46. What is Text Generation?**
> Produce coherent text given prompt. Autoregressive models (GPT).

**47. How do you handle long documents that exceed context length?**
> Truncation, sliding window, hierarchical processing, or use Longformer/BigBird.

**48. What is the maximum context length for BERT?**
> 512 tokens. For GPT-3: 4096. For GPT-4: up to 128K.
