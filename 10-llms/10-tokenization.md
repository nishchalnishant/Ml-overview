---
module: LLMs
topic: Tokenization
subtopic: ""
status: unread
tags: [llms, ml, tokenization]
---
# Tokenization

How text is split into model inputs: algorithms, tradeoffs, failure modes, and interview answers.

**TL;DR:** Tokenization converts raw text into integer IDs. Subword methods (BPE, WordPiece, SentencePiece) dominate because they balance vocabulary size, OOV coverage, and sequence length. Vocab size is a fundamental hyperparameter: too small → high fertility and lost semantics; too large → embedding table memory and sparse softmax. Most LLM bugs involving numbers, code, and non-English text trace back to tokenization.

---

## 1. Why Tokenization Matters

**Input representation:** A model never sees text — it sees a sequence of integer IDs mapped to embeddings. How text is segmented determines what the model can reason about as a unit.

**Vocabulary size → embedding table memory:**
$$\text{embedding bytes} = V \times d_{model} \times \text{dtype\_bytes}$$
At V=100K, d=4096, bf16: 100K × 4096 × 2 = **819 MB** just for the embedding layer (doubled if the output projection is tied).

**OOV (out-of-vocabulary):** Word-level tokenizers map unknown words to a single `<UNK>` token, collapsing all information. Subword methods eliminate true OOV by decomposing into known pieces.

**Fertility:** tokens produced per word (or per Unicode character for non-Latin scripts).
- English with BPE-50K: ~1.3 tokens/word
- Finnish (highly inflected): ~2–3 tokens/word
- Chinese/Japanese with a Latin-trained tokenizer: ~2–4 tokens/character
- High fertility = longer sequences = quadratic attention cost + truncated context.

---

## 2. Character-level vs Word-level vs Subword

| Property | Character-level | Word-level | Subword (BPE/WP/SP) |
|---|---|---|---|
| Vocab size | ~100–500 | 50K–1M+ | 8K–100K (tunable) |
| Sequence length | Very long (5–10× subword) | Short | Medium |
| OOV handling | None (all chars known) | `<UNK>` for rare words | Near-zero OOV |
| Morphological coverage | Full (composable) | Poor | Good |
| Memory (embedding table) | Tiny | Large–huge | Moderate |
| Pretraining compute | High (long seqs) | Moderate | Moderate |
| Arithmetic/code reasoning | Hard (digit-level) | Coarse | Fragmented numbers |
| Multilingual | Natural | Poor | Good with joint vocab |

Character models (e.g. ByT5) are competitive on morphologically rich languages and character-level tasks but suffer from quadratic attention over long sequences.

---

## 3. BPE (Byte-Pair Encoding)

**Origin:** Sennrich et al. 2016 (NMT); popularized for LLMs by GPT-2.

### Algorithm

```
Input: corpus as sequence of characters + </w> end-of-word markers
Target vocab size: V

1. Initialize vocab with all unique characters.
2. Repeat until |vocab| == V:
   a. Count all adjacent symbol pairs in the corpus.
   b. Find the most frequent pair (A, B).
   c. Merge all occurrences of (A, B) → AB in the corpus.
   d. Add AB to vocab; record the merge rule (A B → AB).
3. Output: ordered list of merge rules + final vocab.
```

**Encoding at inference:**

```python
def encode(text, merge_rules):
    tokens = list(text)          # start: one token per char
    for (A, B) in merge_rules:   # apply rules in training order
        i = 0
        while i < len(tokens) - 1:
            if tokens[i] == A and tokens[i+1] == B:
                tokens[i] = A + B
                del tokens[i+1]
            else:
                i += 1
    return tokens
```

Merge rules are applied **in training order** — the first learned merge is applied first.

### Byte-Level BPE (GPT-2 / GPT-4)

Problem: raw BPE over Unicode chars can produce merge rules for rare unicode points; the base vocab explodes.

Solution (GPT-2): map every byte (0–255) to a printable Unicode character → base vocab of exactly **256 symbols**. Merges operate on byte sequences. This eliminates `<UNK>` entirely — any UTF-8 byte sequence is tokenizable.

```
GPT-2 tokenizer: 50,257 tokens  (50,000 merges + 256 byte tokens + <|endoftext|>)
GPT-4 / cl100k_base: 100,277 tokens
```

### BPE Properties
- Greedy, deterministic (given fixed merge rules)
- Same string always maps to same token sequence (no stochasticity)
- No explicit handling of whitespace — GPT-2 BPE encodes leading space as part of the token (`Ġword`)

---

## 4. WordPiece

**Used by:** BERT, DistilBERT, ELECTRA.

### Difference from BPE

BPE merges the **most frequent** pair. WordPiece merges the pair that **maximizes corpus likelihood** under the current unigram language model:

$$\text{score}(A, B) = \frac{\text{count}(AB)}{\text{count}(A) \times \text{count}(B)}$$

This favors merges where the pair is more surprising than its parts — i.e., high pointwise mutual information.

### ## Prefix Convention

Subword pieces that continue a word (not word-initial) are prefixed with `##`:

```
"tokenization" → ["token", "##ization"]
"unbelievable"  → ["un", "##believ", "##able"]
```

The `##` prefix lets the model reconstruct word boundaries from the token sequence.

### BERT Tokenizer

```python
from transformers import BertTokenizer
tok = BertTokenizer.from_pretrained("bert-base-uncased")
tok.tokenize("tokenization")  # ['token', '##ization']
tok.tokenize("COVID-19")      # ['covid', '-', '19']
```

BERT additionally lowercases all input (uncased variant) before tokenization — a lossy step that removes capitalization signal.

---

## 5. SentencePiece

**Used by:** T5, LLaMA, Mistral, mT5, many multilingual models.

**Key design:** operates on raw Unicode text with **no pre-tokenization** (no whitespace split). Whitespace is treated as a normal character; word boundaries are encoded by prepending `▁` (U+2581) to tokens following a space:

```
"Hello world" → ["▁Hello", "▁world"]
"don't"       → ["▁don", "'", "t"]
```

This is **language-agnostic** — no language-specific word-boundary logic needed for Japanese, Chinese, Thai, etc.

### Two SentencePiece Variants

| Variant | Training objective | Notes |
|---|---|---|
| BPE | Same greedy merge as standard BPE | Deterministic |
| Unigram LM | Maximizes corpus likelihood under unigram model; prunes vocab iteratively | Stochastic segmentation possible |

**Unigram LM algorithm:**
1. Start with a large vocabulary (e.g., all substrings up to length 16).
2. Compute unigram probabilities for all pieces.
3. For each piece, compute loss increase if removed.
4. Remove bottom X% by loss impact.
5. Repeat until target vocab size reached.

**Stochastic segmentation** (used during training, not inference): sample from the distribution over valid segmentations rather than taking the argmax. Improves robustness to segmentation artifacts.

---

## 6. Tiktoken

OpenAI's tokenizer library, used for GPT-3.5, GPT-4, and o-series models.

| Encoding | Used by | Vocab size |
|---|---|---|
| `r50k_base` | GPT-3, Codex | 50,257 |
| `p50k_base` | text-davinci-003 | 50,281 |
| `cl100k_base` | GPT-3.5-turbo, GPT-4 | 100,277 |
| `o200k_base` | GPT-4o, o1, o3 | 200,019 |

**Implementation characteristics:**
- Written in Rust with Python bindings → very fast (10–100× faster than pure-Python HuggingFace tokenizers for throughput workloads)
- Uses byte-level BPE under the hood
- Regex-based pre-tokenization to split on whitespace, punctuation, digits — prevents cross-boundary merges
- `cl100k_base` pre-tokenization pattern: splits numbers into individual digits, preventing multi-digit merges that hurt arithmetic

```python
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")
enc.encode("Hello, world!")   # [9906, 11, 1917, 0]
enc.encode("100")             # [1041]  — single token
enc.encode("1000")            # [1000]  — single token
enc.encode("10000")           # [1251, 15, 12279]  — fragmented!
```

---

## 7. Vocab Sizing

### Tradeoffs

| Larger vocab (100K+) | Smaller vocab (8–32K) |
|---|---|
| Lower fertility, shorter sequences | Higher fertility, longer sequences |
| More embedding memory | Less embedding memory |
| Sparser gradient updates (rare tokens undertrained) | Denser gradient updates |
| Better multilingual coverage per token | Non-Latin languages over-fragment |
| More expressive atomic units | More compositional reasoning required |
| Larger softmax → slower output projection | Faster output projection |

### Softmax Bottleneck

At inference, the final logit computation is:
$$\text{logits} = h W_E^T \in \mathbb{R}^V$$

With V=200K and d=4096 (bf16): each token generation requires a 4096×200K matrix-vector product — ~1.6B FLOPs and 1.6 GB of weight just for the output head.

### Typical Sizes

| Model | Tokenizer | Vocab size |
|---|---|---|
| BERT-base | WordPiece | 30,522 |
| GPT-2 | Byte-BPE | 50,257 |
| T5 | SentencePiece BPE | 32,100 |
| LLaMA 2 | SentencePiece BPE | 32,000 |
| LLaMA 3 | Tiktoken-style BPE | 128,256 |
| GPT-4 | cl100k_base | 100,277 |
| GPT-4o | o200k_base | 200,019 |
| mT5 (multilingual) | SentencePiece Unigram | 250,112 |

### Multilingual Coverage

A monolingual English tokenizer applied to other languages causes severe over-fragmentation:
- GPT-2 on Turkish: ~4–6 tokens/word (vs ~1.3 for English)
- GPT-2 on Arabic: ~7–10 tokens/word (right-to-left, rich morphology)

Multilingual models require either a larger shared vocab or per-language sub-vocabularies. mT5 uses 250K tokens to cover 101 languages adequately.

---

## 8. Tokenization Artifacts

### Leading Whitespace

Many tokenizers encode the space before a word as part of the token:
```
"dog"   → different token than " dog"
```
GPT-2: `dog` = 9703, ` dog` = 3290. Passing a word in isolation (e.g., in a few-shot prompt) gives a different embedding than the same word in context.

### Capitalization Sensitivity

`"Paris"` and `"paris"` typically map to different tokens. BERT-uncased lowercases first (losing case info). GPT-style models are case-sensitive — "NATO" and "nato" are distinct tokens.

### Number Tokenization

Numbers fragment unpredictably:
```
"1"    → [16]        "10"   → [940]
"100"  → [1041]      "1000" → [1000]
"9999" → [1484, 19]  "9998" → [1484, 23]
```

The model must learn arithmetic across fragmented representations — "9999" + 1 requires reasoning that "19" in the second token corresponds to the last two digits of 9999. This is a major reason LLMs struggle with multi-digit arithmetic.

Tiktoken's `cl100k_base` mitigates this by splitting all numbers into individual digits during pre-tokenization (regex step), so "1234" → ["1","2","3","4"] → consistent digit tokens.

### Code Tokenization

- Indentation: Python indentation encoded as repeated space tokens; 4 spaces may be 1 token or 4 tokens depending on tokenizer
- Identifiers: `snake_case_variable` often fragments at underscores
- Symbols: `->`, `=>`, `::` may or may not be single tokens
- Comments: tokenized same as code — no structural distinction

### Non-English Fertility

Fertility = tokens / characters (higher = worse):

| Language | GPT-2 fertility | LLaMA 3 fertility |
|---|---|---|
| English | ~0.25 | ~0.22 |
| French | ~0.30 | ~0.25 |
| German | ~0.33 | ~0.27 |
| Turkish | ~0.55 | ~0.35 |
| Arabic | ~0.65 | ~0.40 |
| Chinese | ~0.90 | ~0.45 |
| Thai | ~1.20 | ~0.55 |

LLaMA 3's 128K vocab dramatically improves non-English fertility vs LLaMA 2's 32K vocab.

---

## 9. Impact on Training Dynamics

**Sequence length budget:** Attention is O(S²). Doubling fertility halves usable context at the same compute. A model trained with 4K token context sees ~3K English words but only ~500 high-fertility language words.

**Attention complexity:** Longer sequences from high fertility → more attention steps → increased memory for KV cache → either smaller batches or shorter effective context.

**Arithmetic reasoning:** Fragmented number tokens mean the model must:
1. Map token IDs to digit values (not automatic)
2. Perform carry operations across token boundaries
3. Generalize across inconsistent tokenizations of the same number

Digit-level tokenization (as in cl100k_base's number splitting) reduces this burden by providing a consistent representation.

**Morphological tasks:** Inflections ("run", "runs", "running", "ran") may share token prefixes with BPE/WordPiece, providing implicit morphological structure. Pure word-level tokenizers treat these as completely unrelated.

**Data efficiency:** Higher fertility for rare languages means those languages are "penalized" — the same information requires more tokens and hence more compute budget during pretraining.

---

## 10. Modern Trends

### Character-Aware Tokenization

Models that augment subword embeddings with character-level CNN features (e.g., CharBERT, ELMo). Provides robustness to typos, code-switching, and morphology without sacrificing sequence efficiency.

### MegaByte (Yu et al. 2023)

Tokenizer-free: operates on raw bytes. Uses a hierarchical architecture — local model over bytes within a patch (e.g., 4 bytes), global model over patch representations. Competitive with subword models while being fully byte-level. Eliminates tokenization artifacts entirely.

### CANINE (Clark et al. 2022)

Encodes Unicode codepoints directly. Hashing-based embedding to avoid large vocab tables. Downsamples sequence length with a strided local attention step, then processes a compressed representation with a standard transformer.

### Per-Language Tokenizer Merging

LLaMA 3 approach: train separate BPE tokenizers on monolingual corpora, then merge vocabularies. Allocates token budget proportionally per language. Dramatically reduces fertility for non-English languages while keeping total vocab at 128K.

### Tokenizer-Free via Diffusion

Byte-level diffusion models (e.g., MDLM on bytes) sidestep discrete tokenization entirely by operating in continuous or discrete byte space — still early research.

### Continuous Tokenization for Multimodal

Vision-language models (e.g., LLaVA, Flamingo) learn visual "tokens" via patch embeddings — analogous to subword tokenization but for images. Alignment between visual and text token spaces is an active research area.

---

## 11. Interview Questions

**Q1: Why do LLMs struggle with simple arithmetic like 99 + 1?**

Because numbers tokenize inconsistently. "99" might be a single token; "100" is a different single token — the model never sees explicit digit manipulation during standard forward passes. It must implicitly learn carry rules from token-level patterns, which fails for numbers outside the training distribution. Digit-by-digit tokenization (cl100k_base) and scratchpad prompting both reduce this failure mode.

---

**Q2: What is the difference between BPE and WordPiece?**

Both are greedy subword algorithms but differ in the merge criterion:
- BPE: selects the **most frequent** adjacent pair.
- WordPiece: selects the pair that **maximizes corpus likelihood**, equivalent to choosing pairs with highest PMI: score = count(AB) / (count(A) × count(B)).

WordPiece tends to merge pairs that are mutually informative rather than just common. BPE is used in GPT-2/GPT-4; WordPiece is used in BERT.

---

**Q3: Why does a larger vocabulary help multilingual models?**

A fixed vocabulary must cover all languages. With a small vocab (32K), each language gets a proportionally tiny budget — rare languages are forced to decompose words into many small pieces (high fertility). This means: longer sequences, higher compute cost, and weaker semantic representations for those languages. A 200K+ vocab can give reasonable coverage to dozens of languages simultaneously.

---

**Q4: What is byte-level BPE and why does GPT-2 use it?**

Standard character-level BPE starts with a base vocabulary of all Unicode characters, which can be large and includes many rare codepoints. Byte-level BPE instead maps each byte (0–255) to a printable character, giving a fixed base vocab of exactly 256. All UTF-8 encoded text is thus tokenizable without any `<UNK>`. Merges operate on these byte-level symbols. GPT-2 uses this to guarantee coverage of any input string.

---

**Q5: How does SentencePiece handle whitespace differently from BPE?**

BPE typically pre-tokenizes on whitespace (splits words first, then runs BPE within words). SentencePiece treats whitespace as a normal character and tokenizes the raw string. Leading spaces become part of tokens via the `▁` prefix. This is language-agnostic: languages without whitespace word boundaries (Chinese, Japanese, Thai) are handled without special rules.

---

**Q6: What is "fertility" and why does it matter for non-English languages?**

Fertility is the number of tokens generated per word (or per character). A model has a fixed sequence length budget (e.g., 4K tokens). If English averages 1.3 tokens/word but Arabic averages 7 tokens/word, the Arabic content of a 4K-token context is ~7× shorter in terms of meaningful content. This reduces the effective context window, increases attention cost, and means the model sees fewer training examples' worth of information per token for high-fertility languages.

---

**Q7: If you had to design a tokenizer for a code-heavy LLM, what would you change vs. a standard BPE setup?**

Key changes:
1. **Preserve indentation tokens**: merge spaces specifically so common indentation levels (2, 4, 8 spaces) become single tokens — reduces sequence length for Python/YAML.
2. **Digit-by-digit numbers**: prevent multi-digit merges to make arithmetic and version strings consistent.
3. **Common symbol bigrams**: `->`, `=>`, `::`, `//`, `/*`, `*/` should each be single tokens.
4. **Train on code-heavy corpus**: BPE merge priorities reflect corpus statistics — a code-heavy corpus will naturally promote code-relevant merges.
5. **Preserve identifier boundaries at `_`**: avoid merging across underscores so `snake_case` parts are semantically consistent.

---

## Flashcards

**How does tokenizer fertility vary across languages, and why does high fertility hurt?** #flashcard
Fertility (tokens/word or tokens/char) is ~1.3 tokens/word for English with BPE-50K, ~2–3 for morphologically rich Finnish, and ~2–4 tokens/character for Chinese/Japanese under a Latin-trained tokenizer. High fertility means longer sequences for the same content, which costs more under quadratic attention and effectively truncates the usable context window.

**What guarantees does standard BPE make about determinism, and how does it treat whitespace?** #flashcard
BPE is greedy and deterministic — given fixed merge rules, the same string always maps to the same token sequence. It has no explicit whitespace handling by default; GPT-2's BPE instead folds a leading space into the token itself (e.g. `Ġword`).

**Why is tiktoken fast, and how does its pre-tokenization help arithmetic?** #flashcard
Tiktoken is written in Rust with Python bindings, making it 10–100× faster than pure-Python HuggingFace tokenizers, and uses byte-level BPE under the hood. Its regex-based pre-tokenization splits on whitespace/punctuation/digits before merging, preventing cross-boundary merges — `cl100k_base` specifically splits numbers into individual digits so multi-digit numbers don't fragment unpredictably, which helps arithmetic.

**How much does fertility blow up for non-English, non-Latin-script languages under GPT-2's tokenizer?** #flashcard
GPT-2 tokenizes Turkish at ~4–6 tokens/word (vs ~1.3 for English) and Arabic at ~7–10 tokens/word, since GPT-2's vocabulary was trained overwhelmingly on English text.

**What code-specific tokenization quirks can hurt an LLM's code understanding?** #flashcard
Indentation (repeated spaces may be 1 or 4 tokens depending on tokenizer), identifiers (`snake_case_variable` often fragments at underscores), multi-char symbols (`->`, `=>`, `::` may or may not be single tokens), and comments (tokenized with no structural distinction from code).

**What's the core mechanical difference between BPE's and WordPiece's merge rule?** #flashcard
BPE merges the most frequent adjacent pair. WordPiece merges the pair that maximizes corpus likelihood — equivalent to highest PMI: score = count(AB) / (count(A) × count(B)) — favoring pairs that are mutually informative rather than just common.
