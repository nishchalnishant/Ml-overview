# NLP and Transformers — Interview Reference

---

## 1. Tokenization

Converting raw text into integer IDs the model processes.

### Byte-Pair Encoding (BPE)

Start with individual characters; iteratively merge the most frequent adjacent pair:

```
"low", "lower", "newest", "widest"

Initial vocab: l o w e r n s t i d

Step 1: most frequent pair = "e" + "s" → "es"
Step 2: most frequent pair = "es" + "t" → "est"
...
Final: "low", "low"+"er", "new"+"est", "wid"+"est"
```

GPT-2/GPT-4/LLaMA all use BPE. Vocabulary sizes: GPT-2 = 50,257; LLaMA 3 = 128,256.

### WordPiece (BERT)

Similar to BPE but merges based on likelihood ratio rather than frequency:
$$\text{score}(A, B) = \frac{\text{count}(AB)}{\text{count}(A) \times \text{count}(B)}$$

Subword tokens after the first get a `##` prefix (e.g., `playing` → `play`, `##ing`).

### SentencePiece / Unigram

Builds vocabulary top-down: start with all characters + frequent substrings, prune to target vocabulary size using EM on a unigram language model. Used by T5, LLaMA 1/2, many multilingual models.

### Why subword tokenization?

| Method | Vocab size | OOV handling | Sequence length |
| :--- | :--- | :--- | :--- |
| Word-level | Large (millions) | UNK token | Short |
| Character-level | ~256 | None | Long (4× word) |
| Subword (BPE) | 32k–128k | Always representable | Medium |

---

## 2. Text Representations

### TF-IDF

$$\text{TF-IDF}(t, d) = \underbrace{\frac{f_{t,d}}{\sum_k f_{k,d}}}_{\text{term frequency}} \times \underbrace{\log \frac{N}{|\{d : t \in d\}|}}_{\text{inverse document frequency}}$$

- Sparse vector: vocabulary-sized with mostly zeros
- No semantic similarity: "car" and "automobile" have zero cosine similarity
- Fast and effective for keyword search and bag-of-words classification

### Word Embeddings (Word2Vec, GloVe)

Word2Vec **Skip-gram** objective: given a center word $w_c$, predict context words $w_o$:

$$\mathcal{L} = -\sum_{(c, o) \in \text{corpus}} \log P(w_o \mid w_c) = -\sum \log \frac{\exp(u_o^T v_c)}{\sum_w \exp(u_w^T v_c)}$$

Properties learned: `king - man + woman ≈ queen` (linear relationship between embeddings).

**Limitation:** static embeddings — "bank" has one vector regardless of context (financial vs. river bank).

### Contextual Embeddings (BERT, LLaMA)

Each token's embedding depends on the full surrounding context via Transformer attention. "Bank" in "river bank" and "bank account" get different representations.

---

## 3. Self-Attention — Full Mechanics

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

**Step by step for one token:**
1. Multiply query $q_i$ against all keys: $e_{ij} = q_i \cdot k_j / \sqrt{d_k}$
2. Softmax: $\alpha_{ij} = \exp(e_{ij}) / \sum_j \exp(e_{ij})$
3. Weighted sum of values: $\text{out}_i = \sum_j \alpha_{ij} v_j$

**Why $\sqrt{d_k}$ scaling?**

For random vectors with $d_k$ dimensions, $q \cdot k \sim \mathcal{N}(0, d_k)$ — variance grows with $d_k$. Without scaling:
- Large $d_k$ → large dot products → extreme softmax outputs → vanishing gradients
- $\sqrt{d_k}$ normalizes variance to 1: $\text{Var}(q \cdot k / \sqrt{d_k}) = 1$

**Multi-head attention:** run $h$ heads with $d_k = d_{\text{model}} / h$ each, then project:

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
        # Project and reshape to (B, n_heads, T, d_k)
        q = self.W_q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = self.W_k(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = self.W_v(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) / self.d_k ** 0.5
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.W_o(out)
```

**Attention complexity:** $O(n^2 d)$ — quadratic in sequence length. For $n = 1000$, manageable. For $n = 100{,}000$, Flash Attention's tiling is required.

---

## 4. Positional Encoding

Without positional encoding, Transformer is permutation-invariant — "dog bites man" and "man bites dog" produce the same output.

### Sinusoidal (original Transformer)

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right), \quad PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$

Added to token embeddings. Different frequencies encode different scales of position.

### Rotary Position Embedding (RoPE)

Used by LLaMA, GPT-NeoX, Mistral. Encodes relative positions in Q/K via rotation:

$$f_q(x_m, m) = R_m x_m, \quad f_k(x_n, n) = R_n x_n$$

$$q_m \cdot k_n = (R_m x_m)^T (R_n x_n) = x_m^T R_m^T R_n x_n = x_m^T R_{n-m} x_n$$

The dot product depends only on the relative position $n - m$, not absolute positions. Better generalization to unseen context lengths.

---

## 5. BERT vs GPT — Full Comparison

| Dimension | BERT (Encoder) | GPT (Decoder) |
| :--- | :--- | :--- |
| **Architecture** | Encoder-only | Decoder-only |
| **Attention** | Bidirectional (sees all tokens) | Causal (left-to-right only) |
| **Pretraining** | MLM + NSP | Next-token prediction |
| **Fine-tuning** | Add task head, train on labels | Prompt engineering or SFT |
| **Best for** | Classification, NER, QA | Generation, chat, code |
| **Token access** | Both directions | Past tokens only |
| **Example models** | BERT, RoBERTa, DeBERTa | GPT-4, LLaMA, Mistral |

### Masked Language Modeling (BERT)

Randomly mask 15% of tokens; predict the masked tokens:

```
Input:  "The [MASK] sat on the [MASK]"
Target: "cat" at position 2, "mat" at position 6
```

$$\mathcal{L}_{\text{MLM}} = -\sum_{i \in \text{masked}} \log P_\theta(x_i \mid x_{\backslash i})$$

Bidirectional context: the model sees both "The" and "sat on the mat" to predict "cat".

### Causal Language Modeling (GPT)

Predict each token given all previous tokens:

$$\mathcal{L}_{\text{CLM}} = -\sum_{t=1}^T \log P_\theta(x_t \mid x_1, \ldots, x_{t-1})$$

Masked attention ensures position $t$ cannot see positions $> t$ during training.

---

## 6. Encoder-Decoder Transformers (T5, BART)

**Cross-attention:** decoder attends to encoder outputs:

$$\text{CrossAttn}(Q_{\text{dec}}, K_{\text{enc}}, V_{\text{enc}}) = \text{softmax}\left(\frac{Q_{\text{dec}} K_{\text{enc}}^T}{\sqrt{d_k}}\right) V_{\text{enc}}$$

Each decoder step queries what encoder context is most relevant for the next output token.

**When to use encoder-decoder:**
- Translation: encode source language → decode target language
- Summarization: encode long document → decode summary
- Seq2seq tasks where input and output are in different spaces

**T5 framing:** every task is "text-to-text" — classification becomes "classify: [text] → positive/negative", making a single model handle all tasks.

---

## 7. Key Transformer Improvements

| Technique | Change | Effect |
| :--- | :--- | :--- |
| **Pre-norm** (GPT-2) | LayerNorm before attention, not after | More stable training |
| **RoPE** (LLaMA) | Relative positional encoding | Better length generalization |
| **SwiGLU** (LLaMA) | Gated activation in FFN | Better performance |
| **GQA** (LLaMA 3) | Grouped-query attention | Smaller KV cache |
| **Flash Attention** | IO-aware tiling | Same math, 2-4× faster |
| **RMS Norm** (LLaMA) | Skip mean subtraction | Faster normalization |

### SwiGLU (LLaMA feed-forward)

Standard FFN: $\text{FFN}(x) = \text{ReLU}(xW_1 + b_1) W_2 + b_2$

SwiGLU: $\text{FFN}(x) = (\text{Swish}(xW_1) \odot xW_3) W_2$

where $\text{Swish}(x) = x \cdot \sigma(x)$. The element-wise gating (third weight matrix $W_3$) allows the network to selectively pass information, empirically improving performance.

---

## 8. PEFT Methods

Fine-tune large models by training only a small fraction of parameters.

### LoRA (Low-Rank Adaptation)

For a weight matrix $W \in \mathbb{R}^{d \times k}$, add a low-rank update:

$$W' = W + \Delta W = W + BA$$

where $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$, rank $r \ll \min(d, k)$.

**Parameter reduction:** original $d \times k$ params → only $r(d + k)$ params. For $d = k = 4096$, $r = 16$: $16M \to 128k$ (125× reduction).

```python
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=16,                   # rank
    lora_alpha=32,          # scaling: effective lr = lora_alpha / r
    target_modules=["q_proj", "v_proj"],  # which weight matrices
    lora_dropout=0.1,
    bias="none",
)

model = get_peft_model(base_model, config)
model.print_trainable_parameters()
# trainable params: 4,194,304 || all params: 6,742,609,920 || trainable%: 0.0623
```

### Prefix-Tuning

Prepend $k$ trainable tokens (the "prefix") to keys and values in every layer. The prefix acts like a soft prompt that steers the model's attention patterns.

**Downside:** reduces effective context window by $k$ tokens per layer.

### Comparison

| Method | Trainable params | Inference overhead | Context window |
| :--- | :--- | :--- | :--- |
| Full fine-tuning | 100% | None | Unchanged |
| LoRA | ~0.1% | None (merge weights) | Unchanged |
| Prefix-tuning | ~0.1% | Small | Reduced by prefix length |
| Prompt tuning | <0.01% | None | Reduced by prompt tokens |
| Adapter layers | ~1% | Small (serial layers) | Unchanged |

---

## 9. Evaluation Metrics for NLP

| Task | Metric | Formula / Notes |
| :--- | :--- | :--- |
| Classification | Accuracy, F1 | $F1 = 2 \cdot \frac{P \cdot R}{P + R}$ |
| Generation | BLEU | n-gram precision vs. reference |
| Summarization | ROUGE-L | LCS recall between generated and reference |
| Language modeling | Perplexity | $\exp\left(-\frac{1}{T}\sum_t \log P(x_t \mid x_{<t})\right)$ |
| Embedding quality | MTEB benchmark | Downstream task performance across 56 tasks |
| Chat/instruction | MT-Bench, AlpacaEval | LLM-as-judge pairwise comparison |

**Perplexity intuition:** a perplexity of $k$ means the model is on average as confused as if choosing uniformly among $k$ tokens. Lower is better; GPT-2 small: ~29 on WikiText-103; GPT-4: ~8.

> [!TIP]
> **Interview structure:** NLP = tokenization (BPE handles rare words) → representation (static word2vec → contextual BERT/GPT) → attention (scaled dot-product, $\sqrt{d_k}$ for gradient stability) → architecture choice (encoder for understanding, decoder for generation, encoder-decoder for seq2seq). BERT bidirectionality requires MLM training — can't do causal generation. GPT's causal mask enables generation but limits to left context only.
