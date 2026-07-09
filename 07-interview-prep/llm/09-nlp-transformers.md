---
module: Interview Prep
topic: Llm
subtopic: Nlp Transformers
status: unread
tags: [interviewprep, ml, llm-nlp-transformers]
---
# NLP and Transformers — First-Principles Interview Guide

> For classical-NLP-to-Transformers coverage from the core ML track, see [ml/13-nlp.md](../ml/13-nlp.md).

---

## 1. Tokenization

### The problem tokenization solves

A neural network processes numbers, not text. The question is: what granularity should the numbers represent?

**Word-level:** treat each word as a token. Problem: vocabulary size is unbounded (technical papers, proper nouns, code). Seeing "unrecognizable" at test time — a word not in the training vocabulary — produces a dead UNK token. The model knows nothing about the word even though it could decompose it into "un-" + "recognizable" + "-izable."

**Character-level:** treat each character as a token. Problem: sequences become 4× longer, diluting the model's effective context window. "Hello world" is 2 word tokens or 11 character tokens. For a 4K context limit, character-level effectively gives you ~1K words of context.

**Subword:** the resolution. Decompose words into pieces, where common words stay whole and rare words split into recognizable components. "Unrecognizable" → "un" + "recogniz" + "able" — meaningful pieces that the model has seen in other contexts.

### BPE (Byte-Pair Encoding)

Used by GPT-2, GPT-4, LLaMA 3.

Start with the character vocabulary. Iteratively merge the most frequent adjacent pair:

```
Corpus: "low", "lower", "lowest", "newer", "newest"

Step 1: Count pair frequencies
        "e"+"s" appears 2x → merge to "es"

Step 2: Count again
        "es"+"t" appears 2x → merge to "est"

Final: "low", "low"+"er", "low"+"est", "new"+"er", "new"+"est"
```

Repeat until you hit the target vocabulary size. GPT-2 = 50,257 tokens; LLaMA 3 = 128,256 tokens.

### WordPiece (BERT)

Same idea as BPE but merges the pair that maximizes the likelihood of the training corpus, not just the most frequent pair:
$$\text{score}(A, B) = \frac{\text{count}(AB)}{\text{count}(A) \times \text{count}(B)}$$

This selects pairs that, when merged, most improve the language model likelihood — not just the pairs that appear most often. Subword tokens after the first get `##` prefix: "playing" → "play" + "##ing".

### SentencePiece / Unigram

Top-down approach: start with all possible substrings, then prune. Assign each possible token a log probability under a unigram language model, and iteratively remove tokens whose removal minimally hurts the model likelihood (EM algorithm). Used by T5, LLaMA 1/2.

**Comparison:**

| Method | OOV handling | Sequence length | Vocab control |
| :--- | :--- | :--- | :--- |
| Word-level | UNK = information loss | Shortest | Hard — can't limit |
| Character-level | None | ~4× longer | Fixed at ~256 |
| BPE | Always representable | Medium | Exact target size |

---

## 2. Text Representations

### The problem contextual embeddings solve

Static word embeddings (Word2Vec, GloVe) assign one vector per word. "Bank" in "river bank" and "bank account" get the same representation. The model must disambiguate from context, but the representation itself carries no context.

### Word2Vec (Skip-gram)

Given a center word, predict surrounding context words. The model learns that words appearing in similar contexts have similar vectors — "king" and "queen" appear near "royal", "palace", "throne", so their vectors become similar.

Training objective:
$$\mathcal{L} = -\sum_{(c,o) \in \text{corpus}} \log P(w_o \mid w_c) = -\sum \log \frac{\exp(u_o^T v_c)}{\sum_w \exp(u_w^T v_c)}$$

**The famous property:** "king" - "man" + "woman" ≈ "queen". Linear arithmetic works because the vector space encodes relational differences: the vector `man → king` is similar to `woman → queen`.

**The limitation:** static. Once trained, "bank" has one vector. Cannot distinguish "the bank was robbed" from "on the river bank." This is the problem contextual embeddings solve.

### TF-IDF

$$\text{TF-IDF}(t, d) = \frac{f_{t,d}}{\sum_k f_{k,d}} \times \log \frac{N}{|\{d : t \in d\}|}$$

Sparse, vocabulary-sized vector. Weights tokens by how often they appear in this document relative to how common they are across all documents. "The" appears in every document → low IDF → downweighted. "Hepatocyte" appears rarely → high IDF → upweighted.

No semantic understanding: "car" and "automobile" have zero cosine similarity. But fast, works without labeled data, strong baseline for keyword search and bag-of-words classification.

---

## 3. Self-Attention — The Mechanism and Why Each Part Exists

### The problem self-attention solves

RNNs process sequences token by token. The hidden state at position $t$ must somehow encode everything relevant from positions $1, ..., t-1$ through a fixed-size vector. For long sequences, early information gets overwritten. The path length between any two positions is proportional to their distance.

Self-attention collapses this to $O(1)$: every position can directly attend to every other position in a single operation. The model doesn't need to "remember" — it can look up.

### The mechanism, derived from first principles

**Query, Key, Value — the database analogy:**
- Query ($Q$): what am I looking for?
- Key ($K$): what do I advertise?
- Value ($V$): what is my actual content?

For each token $i$, compute compatibility with all tokens $j$: $e_{ij} = q_i \cdot k_j$. Softmax over $j$ gives attention weights $\alpha_{ij}$. Output is a weighted sum of values: $\text{out}_i = \sum_j \alpha_{ij} v_j$.

**Why $\sqrt{d_k}$ scaling:** Without it, dot products have variance $d_k$; at $d_k=64$ scores have std 8, causing softmax to collapse to a one-hot vector and zeroing gradients. Dividing by $\sqrt{d_k}$ restores variance to 1. Full derivation: [math-derivations.md §5](../ml/18-math-derivations.md#5-why-sqrtd_k-scaling).

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

**Why multiple heads:**

A single attention head computes one set of Q/K/V projections — one way of looking at the sequence. Different aspects require different attention patterns: one head might capture subject-verb agreement (syntactic), another coreference resolution (semantic), another local context (positional). Multi-head attention runs $h$ attention operations in parallel, each learning different projections:

$$\text{MHA}(Q,K,V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O$$
$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

Each head uses $d_k = d_{\text{model}}/h$ — same total compute as one large head.

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        B, T, C = x.shape
        # Project and split into heads: (B, T, C) → (B, n_heads, T, d_k)
        q = self.W_q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = self.W_k(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = self.W_v(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) / self.d_k**0.5  # (B, n_heads, T, T)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.W_o(out)
```

**Attention complexity:** $O(n^2 d)$ compute, $O(n^2)$ memory for the attention matrix. At $n = 1{,}000$: 1M entries, manageable. At $n = 100{,}000$: 10B entries — cannot materialize in GPU HBM. Flash Attention tiles the computation to stay in SRAM, producing the same result without materializing the full matrix.

---

## 4. Positional Encoding — Why It Exists

### The problem: Transformers are permutation-invariant

Self-attention is a set operation. The output for token $i$ depends on which tokens are present, but not on their order. "Dog bites man" and "man bites dog" produce identical attention scores (different tokens, but the same *set* of (Q, K, V) interactions up to permutation). Without positional information, these sentences are indistinguishable.

**The consequence:** a language model without positional encoding cannot learn that word order matters — it would assign the same probability to both sentences. This is catastrophically wrong for language.

### Sinusoidal (original Transformer)

Add position-dependent signals directly to token embeddings:
$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right), \quad PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$

Different frequencies encode different scales: high-frequency dimensions distinguish nearby positions; low-frequency dimensions distinguish far-apart positions. The original paper argued that relative positions can be expressed as linear functions of absolute positions under this scheme.

**Limitation:** fixed frequencies, no learned parameters. The model can generalize to positions it hasn't seen (longer sequences at inference), but the positional signal isn't optimized for the task.

### RoPE (Rotary Position Embedding)

Used by LLaMA, Mistral, GPT-NeoX. Encodes position by rotating query and key vectors:
$$f_q(x_m, m) = R_m x_m, \quad f_k(x_n, n) = R_n x_n$$

The dot product in attention:
$$q_m \cdot k_n = (R_m x_m)^T (R_n x_n) = x_m^T R_m^T R_n x_n = x_m^T R_{m-n} x_n$$

The attention score depends only on $m - n$ (the relative distance), not on the absolute positions $m$ and $n$ separately. This provides a strong inductive bias: position 100 vs 101 should feel similar to position 200 vs 201. Empirically, RoPE generalizes better to sequence lengths not seen during training than sinusoidal or learned absolute embeddings.

---

## 5. BERT vs GPT — The Architecture Split

### Why two different architectures exist

The pretraining objective determines the architecture.

If you want a model that can **generate** text — produce one token at a time, conditioning on all previous tokens — the model must not see future tokens. This requires a causal (autoregressive) mask. Any architecture where position $t$ can see position $t+1$ during training is cheating: it learns to "predict" the next token by reading it.

If you want a model that produces **rich representations** of text — where each token's embedding captures its full context — the model should see both directions. Why would you artificially hide the right context when building a representation? BERT's bidirectional attention lets each token attend to all other tokens, producing richer representations for classification and understanding.

| Dimension | BERT (Encoder) | GPT (Decoder) |
| :--- | :--- | :--- |
| Attention mask | Bidirectional (all positions visible) | Causal (only past positions) |
| Pretraining | Masked Language Modeling (MLM) | Next-token prediction (CLM) |
| What it optimizes | Filling in masked tokens using full context | Predicting the next token from left context |
| Best for | Classification, NER, extractive QA, embeddings | Text generation, chat, code, reasoning |
| Can it generate? | No — future tokens visible during training | Yes |
| Can it produce bidirectional representations? | Yes | No — each position only sees past |

### Masked Language Modeling (BERT)

Mask 15% of tokens; predict them from the rest:
```
Input:  "The [MASK] sat on the [MASK]"
Target: "cat" and "mat"
```
$$\mathcal{L}_{\text{MLM}} = -\sum_{i \in \text{masked}} \log P_\theta(x_i \mid x_{\backslash i})$$

The model sees the full non-masked context in both directions. This forces each token to be predicted from rich contextual information, producing embeddings that encode meaning rather than position.

### Causal Language Modeling (GPT)

Predict each token given all previous tokens:
$$\mathcal{L}_{\text{CLM}} = -\sum_{t=1}^T \log P_\theta(x_t \mid x_1, \ldots, x_{t-1})$$

The causal mask (upper triangular −∞ before softmax) enforces that position $t$ cannot see positions $> t$. This makes GPT naturally a generation model — at inference, you have exactly the same setup as training: generate token $t$ from tokens $1, ..., t-1$.

---

## 6. Encoder-Decoder Transformers (T5, BART)

### When you need both

Translation and summarization are structurally different from classification or generation. You have a long input (source text) and a structured output (translation / summary) that must capture the full semantics of the input while being in a different form.

A decoder-only model must "remember" everything about the source within its own autoregressive context — it can't maintain a separate rich representation of the input. An encoder-decoder provides the encoder to build a rich bidirectional representation of the input, then the decoder generates the output, attending to that representation through cross-attention.

**Cross-attention:**
$$\text{CrossAttn}(Q_{\text{dec}}, K_{\text{enc}}, V_{\text{enc}}) = \text{softmax}\left(\frac{Q_{\text{dec}} K_{\text{enc}}^T}{\sqrt{d_k}}\right) V_{\text{enc}}$$

Decoder queries say "what encoder context is relevant for the next output token?" Keys and values come from the encoder. Each generation step can selectively attend to different parts of the source.

**T5's unification:** every NLP task framed as text-to-text:
- Classification: "classify sentiment: This movie was great. → positive"
- Translation: "translate English to French: Hello world. → Bonjour monde."
- QA: "question: What is the capital? context: France is in Europe. → Paris"

Same model, same loss, same tokenizer — the task prefix tells the model what to do.

---

## 7. Key Transformer Improvements — What Problem Each Solved

| Technique | The problem it solved | Effect |
| :--- | :--- | :--- |
| Pre-norm (GPT-2) | Post-norm layers caused gradient explosion at initialization | Stable training from random init |
| RoPE (LLaMA) | Sinusoidal PE doesn't generalize to unseen lengths | Better extrapolation beyond training context |
| SwiGLU (LLaMA) | Standard FFN's ReLU gates information crudely | Selective gating improves quality |
| GQA (LLaMA 2/3) | KV cache grew linearly with attention heads | Smaller KV cache, faster decoding |
| Flash Attention | $O(n^2)$ memory prevented long contexts | Same math, constant memory, 2–4× faster |
| RMSNorm (LLaMA) | LayerNorm's mean subtraction adds compute without much benefit | Faster normalization |

**SwiGLU in detail:**

Standard FFN: $\text{FFN}(x) = \text{ReLU}(xW_1 + b_1) W_2$

SwiGLU: $\text{FFN}(x) = (\text{Swish}(xW_1) \odot xW_3) W_2$, where $\text{Swish}(x) = x \sigma(x)$

The element-wise product with $xW_3$ provides a gating mechanism: the network can suppress parts of the intermediate representation that aren't useful for the current token. This improves performance without changing model size.

**Grouped-Query Attention (GQA):**

Standard MHA: $H$ heads, each with its own $K$ and $V$ projections. KV cache grows as $O(H \cdot d_h \cdot n)$.

GQA: $G$ groups of heads share $K$ and $V$. KV cache shrinks by $H/G$ factor. LLaMA 3 70B uses $G = 8$ groups with $H = 64$ heads — 8× smaller KV cache than standard MHA. This directly increases the number of requests that can be batched in GPU memory.

---

## 8. PEFT Methods — Parameter-Efficient Fine-Tuning

### The problem PEFT solves

Full fine-tuning a 70B model requires storing 70B × 4 bytes = 280GB of gradients and optimizer state in addition to the model weights. This is unaffordable for most settings and risks catastrophic forgetting (the model overwrites pretraining knowledge with fine-tuning examples).

PEFT methods freeze most parameters and train only a small, targeted subset. The insight: the weight *update* needed for a new task lives in a low-dimensional subspace of the full parameter space.

### LoRA (Low-Rank Adaptation)

For a weight matrix $W \in \mathbb{R}^{d \times k}$, add a low-rank decomposition:
$$W' = W + \Delta W = W + BA, \quad B \in \mathbb{R}^{d \times r},\; A \in \mathbb{R}^{r \times k}$$

Only $A$ and $B$ are trained. Rank $r$ is typically 4–64 — far smaller than $\min(d, k)$.

**Parameter reduction:** for $d = k = 4096$, $r = 16$:
- Full matrix: $4096^2 = 16.8M$ parameters
- LoRA: $16 \times (4096 + 4096) = 131K$ parameters — 128× reduction

**At inference:** merge the update: $W' = W + BA$. Zero inference overhead.

**Scaling:** $\Delta W = \frac{\alpha}{r} BA$, where $\alpha$ is a scaling hyperparameter. Setting $\alpha = 2r$ keeps the update magnitude independent of rank.

```python
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=16,
    lora_alpha=32,                              # scaling: alpha/r = 2.0
    target_modules=["q_proj", "v_proj"],        # which weight matrices to adapt
    lora_dropout=0.1,
    bias="none",
)
model = get_peft_model(base_model, config)
model.print_trainable_parameters()
# trainable params: 4,194,304 || all params: 6,742,609,920 || trainable%: 0.0623
```

**Why q_proj and v_proj?** Attention query and value projections capture how the model attends and what it retrieves. These are the weights most task-specific — more so than the MLP layers that process the attended information.

### Comparison

| Method | Trainable | Inference cost | Context window |
| :--- | :--- | :--- | :--- |
| Full fine-tuning | 100% | None | Unchanged |
| LoRA | ~0.1% | None (merge at inference) | Unchanged |
| Prefix-tuning | ~0.1% | Small (extra tokens) | Reduced by prefix length |
| Prompt tuning | < 0.01% | None | Reduced by prompt tokens |

---

## 9. NLP Evaluation Metrics

| Task | Primary metric | Why not just accuracy |
| :--- | :--- | :--- |
| Classification | F1, AUC-ROC | Imbalanced classes; accuracy misleads |
| Text generation | BLEU | N-gram precision vs reference; coarse but standard |
| Summarization | ROUGE-L | LCS-based recall; measures coverage |
| Language modeling | Perplexity | Model confidence per token; model-comparable |
| Embedding quality | MTEB benchmark | Downstream task performance across 56 tasks |
| Chat / instruction following | MT-Bench, AlpacaEval | LLM-as-judge pairwise comparison |

**Perplexity — the intuition:**
$$\text{Perplexity} = \exp\left(-\frac{1}{T}\sum_{t=1}^T \log P(x_t \mid x_{<t})\right)$$

Perplexity $k$ means the model is, on average, as uncertain as if choosing uniformly among $k$ tokens at each step. Lower is better. GPT-2 (124M params) on WikiText-103: ~29. GPT-4: ~8. The improvement reflects both scale and training data quality.

**BLEU's known limitations:**
- Penalizes paraphrases that use different but equivalent words
- Sentence-level BLEU is unreliable; use corpus-level
- High BLEU doesn't imply fluency or factual accuracy
- Now largely replaced by neural metrics (BERTScore) for research, but still ubiquitous in industry baselines

**Why ROUGE-L for summarization:**
LCS (Longest Common Subsequence) between generated and reference summary. Captures key phrases in order without requiring exact contiguous n-grams. More robust than ROUGE-1/2 for paraphrased summaries.
