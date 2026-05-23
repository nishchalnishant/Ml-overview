---
module: References
topic: Book Notes
subtopic: Deep Learning Build A Large Language Model From Scratch
status: unread
tags: [references, ml, book-notes-deep-learning]
---
# Build a Large Language Model From Scratch

## Chapter 1: Understanding Large Language Models

**The problem the book is addressing**
Most practitioners use LLMs as black boxes via APIs. When something goes wrong — hallucinations, poor instruction following, domain failures — they have no mental model for *why*. Building one from scratch forces understanding of every design decision.

**The core insight**
LLMs are decoder-only transformers trained on next-token prediction. The apparent complexity (translation, reasoning, code) emerges from this single objective applied at scale. The architecture is simpler than it looks — the book's GPT implementation is ~300 lines of PyTorch.

**The mechanics**
- Two-phase development: pretraining (language modeling on unlabeled text) → fine-tuning (task-specific with labels or RLHF)
- GPT uses only the transformer decoder — no encoder, no cross-attention
- Emergent capabilities (zero-shot, few-shot) arise from pretraining; not explicitly trained
- Custom LLMs justify their cost when: data is private, latency is critical, or domain specificity matters

**What the book gets right / what to watch out for**
Building from scratch is the right learning approach. In production, you don't build from scratch — you fine-tune existing open-weight models (LLaMA, Mistral). The book's GPT is trained on small corpora for illustration; production pretraining requires 100B+ tokens and multi-GPU clusters.

---

## Chapter 2: Working with Text Data

**The problem the book is addressing**
Neural networks require fixed-size numerical inputs. Text is variable-length symbolic sequences. Naive tokenization (character-level) creates very long sequences; word-level tokenization can't handle out-of-vocabulary words. You need a principled middle ground.

**The core insight**
Byte Pair Encoding (BPE) learns a subword vocabulary by iteratively merging the most frequent adjacent symbol pairs. Common words become single tokens; rare words decompose into meaningful subwords. This balances vocabulary size with sequence length and handles any input without OOV issues.

**The mechanics**
- BPE training: start with character vocabulary; repeatedly merge most frequent adjacent pair; stop at target vocab size (50k typical for GPT-2)
- Encoding: greedily apply learned merges to new text
- Special tokens: `<|endoftext|>` marks document boundaries; `<|unk|>` rarely needed with BPE
- Token embedding: map each integer token ID to a d-dimensional vector via a lookup table (trained)

**What the book gets right / what to watch out for**
BPE is the right choice for LLMs. One pitfall: BPE is language-aware at training time — GPT-2's tokenizer is optimized for English and inefficient for other scripts (Chinese, Arabic each use 2–3 tokens per character). Tiktoken (OpenAI's fast BPE implementation) is the production choice. Never train your own tokenizer unless you're pretraining from scratch on a new domain/language.

---

## Chapter 3: Coding Attention Mechanisms

**The problem the book is addressing**
RNNs process sequences step-by-step — they can't parallelize and struggle with long-range dependencies. How do you build a mechanism that lets every token attend to every other token in parallel?

**The core insight**
Self-attention computes a weighted sum of value vectors, where weights come from the compatibility of query and key vectors. For autoregressive LLMs, causal masking prevents attending to future tokens. The entire attention computation is parallelizable across the sequence dimension.

**The mechanics**
- Simplified self-attention (no parameters): A = softmax(X Xᵀ / √d) · X
- Trainable self-attention: Q=XW_Q, K=XW_K, V=XW_V; A = softmax(QKᵀ/√d_k)·V
- Causal mask: set upper-triangle of QKᵀ to -inf before softmax
- Multi-head: run h heads with d_k = d_model/h; concatenate outputs; project with W_O

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
    def forward(self, x):
        B, T, C = x.shape
        Q = self.W_q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        scores = Q @ K.transpose(-2, -1) / self.d_k**0.5
        # causal mask
        mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
        scores.masked_fill_(mask, float('-inf'))
        attn = scores.softmax(-1)
        out = (attn @ V).transpose(1, 2).contiguous().view(B, T, -1)
        return self.W_o(out)
```

**What the book gets right / what to watch out for**
The from-scratch implementation builds real intuition. In production, use `F.scaled_dot_product_attention` (PyTorch 2.0) or FlashAttention — they are fused CUDA kernels that are 2–4× faster and use O(n) memory instead of O(n²). Never write naive attention for sequences longer than 2k tokens.

---

## Chapter 4: Implementing a GPT Model from Scratch

**The problem the book is addressing**
The transformer block is the core repeating unit of GPT but the design choices (Pre-LayerNorm vs Post-LayerNorm, GELU vs ReLU, feed-forward expansion factor) are often cargo-culted without understanding their purpose.

**The core insight**
A GPT transformer block is: LayerNorm → MultiHeadAttention → residual add → LayerNorm → FFN → residual add. Pre-LayerNorm (applied before the sublayer, not after) is crucial for training stability in deep models. The FFN uses a 4× expansion, allowing each token to process information in a higher-dimensional space independently.

**The mechanics**
```python
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = MultiHeadAttention(cfg.d_model, cfg.n_heads, cfg.dropout)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model),
            nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model),
            nn.Dropout(cfg.dropout)
        )
    def forward(self, x):
        x = x + self.attn(self.ln1(x))   # pre-norm residual
        x = x + self.ffn(self.ln2(x))    # pre-norm residual
        return x

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.context_length, cfg.d_model)
        self.blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
```

**What the book gets right / what to watch out for**
Pre-LayerNorm is correct for training stability — the original transformer used Post-LN which required careful learning rate warmup. Weight tying (tok_emb.weight == head.weight) reduces parameters and is standard in GPT-2/3. The GELU activation is correct for modern LLMs; ReLU is slightly worse. Bias=False in the final projection is now standard (GPT-NeoX, LLaMA).

---

## Chapter 5: Pretraining on Unlabeled Data

**The problem the book is addressing**
You have a model architecture. How do you train it? The training loop for LLMs has several non-obvious details: gradient accumulation, learning rate scheduling, checkpoint management, and evaluation during training.

**The core insight**
Pretraining is next-token prediction: given tokens [t₁,...,tₙ], predict [t₂,...,tₙ₊₁]. The loss is cross-entropy averaged over all positions. The model sees the entire training corpus multiple times (epochs), but typically each epoch is ~1 pass through a massive dataset.

**The mechanics**
```python
def calc_loss(model, input_ids, target_ids):
    logits = model(input_ids)          # (B, T, vocab_size)
    loss = F.cross_entropy(logits.flatten(0,1), target_ids.flatten())
    return loss

def train_step(model, optimizer, input_ids, target_ids):
    optimizer.zero_grad()
    loss = calc_loss(model, input_ids, target_ids)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # gradient clipping
    optimizer.step()
    return loss.item()
```
- Cosine LR schedule with warmup: linearly increase to peak LR over first 2% of steps, then cosine decay
- AdamW optimizer: Adam + weight decay on all non-bias/non-norm parameters
- Evaluate perplexity on held-out validation text every N steps
- Save checkpoints; load GPT-2 weights with `model.load_state_dict(torch.load(...))`

**What the book gets right / what to watch out for**
Gradient clipping (norm ≤ 1.0) is essential — without it, occasional large gradients destabilize training. Cosine decay is standard and outperforms fixed LR. The book's training run is illustrative; real pretraining runs for weeks on thousands of GPUs. BFloat16 mixed precision is standard in production (more range than float16, less memory than float32).

---

## Chapter 6: Fine-Tuning for Classification

**The problem the book is addressing**
A pretrained LLM knows language but not specific tasks. How do you adapt it for classification (sentiment, spam detection) without forgetting what it learned during pretraining?

**The core insight**
Replace the language modeling head with a classification head on top of the final layer's [last token] representation. Fine-tune the entire model (or just the head + last few layers) on labeled examples. The pretrained weights provide a strong initialization.

**The mechanics**
- Remove LM head; add `nn.Linear(d_model, num_classes)`
- Forward pass: take hidden state at last token position → classification logits
- Loss: cross-entropy on class labels
- Selective freezing: freeze all transformer blocks, train only the head (faster, less data needed, more risk of underfitting)
- Full fine-tuning: unfreeze all parameters (slower, needs more data, usually better)

**What the book gets right / what to watch out for**
The "last token" approach is specific to GPT (causal models — the last token has attended to all previous ones). For BERT-style models, use the [CLS] token instead. Fine-tuning the entire model on small datasets can overfit — use early stopping based on validation loss. LoRA (next chapter) is often better than full fine-tuning for small datasets.

---

## Chapter 7: Fine-Tuning to Follow Instructions

**The problem the book is addressing**
A pretrained LLM completes text — it doesn't follow instructions. Ask "What is the capital of France?" and it continues the sentence in unexpected ways. How do you teach a model to respond helpfully to instructions?

**The core insight**
Instruction fine-tuning: train on (instruction, response) pairs in a specific prompt format. The model learns to recognize the format and generate appropriate responses. RLHF adds a second stage where a reward model (trained on human preference data) guides further fine-tuning via PPO.

**The mechanics**
- Format: `<|system|>You are a helpful assistant.<|user|>{instruction}<|assistant|>{response}`
- Supervised fine-tuning (SFT): compute loss only on the response tokens (mask instruction tokens)
- RLHF pipeline:
  1. Train reward model on human comparisons (preferred vs rejected responses)
  2. Use PPO to maximize reward model score while staying close to SFT model (KL penalty)
- LoRA for efficient fine-tuning: freeze base model; add low-rank matrices ΔW = A·B to attention projections; train only A and B

**What the book gets right / what to watch out for**
SFT alone is often sufficient for instruction following — RLHF adds alignment quality but is complex to implement correctly (reward hacking is a real risk). LoRA is the practical choice for fine-tuning: 10–100× fewer trainable parameters, same GPU memory. DPO (Direct Preference Optimization) has largely replaced PPO in practice — it's simpler and avoids the reward model training step.

---

## Appendix: PyTorch and Training Infrastructure

**The problem the book is addressing**
Understanding transformer code requires fluency with PyTorch's core abstractions. Subtle bugs in tensor shapes, gradient flow, or device placement cause training failures that are hard to debug without understanding the underlying mechanics.

**The core insight**
PyTorch's execution model is eager by default (immediate execution, easy debugging) with optional JIT compilation for performance. The autograd engine records operations on tensors with `requires_grad=True` and computes gradients via reverse-mode autodiff when `.backward()` is called.

**The mechanics**
- `nn.Module`: base class for all models; `.parameters()` yields all trainable tensors; `.train()/.eval()` switches dropout/BN behavior
- `DataLoader`: wraps a `Dataset`; handles batching, shuffling, multi-process loading
- `optimizer.zero_grad()` → `loss.backward()` → `optimizer.step()`: standard training loop
- `torch.no_grad()`: context manager that disables gradient tracking (inference, evaluation)
- Model loading: `model.load_state_dict(state_dict, strict=False)` allows partial loading (e.g., loading GPT-2 weights into custom architecture)

**What the book gets right / what to watch out for**
The PyTorch appendix correctly identifies the common pitfall of forgetting `optimizer.zero_grad()` — gradients accumulate by default. For production training, add `torch.compile(model)` (PyTorch 2.0) for 30–50% speedup without code changes. Mixed precision (`torch.autocast`) is essential for GPU memory efficiency.

## Flashcards

**Two-phase development?** #flashcard
pretraining (language modeling on unlabeled text) → fine-tuning (task-specific with labels or RLHF)

**GPT uses only the transformer decoder?** #flashcard
no encoder, no cross-attention

**Emergent capabilities (zero-shot, few-shot) arise from pretraining; not explicitly trained?** #flashcard
Emergent capabilities (zero-shot, few-shot) arise from pretraining; not explicitly trained

**Custom LLMs justify their cost when?** #flashcard
data is private, latency is critical, or domain specificity matters

**BPE training?** #flashcard
start with character vocabulary; repeatedly merge most frequent adjacent pair; stop at target vocab size (50k typical for GPT-2)

**Encoding?** #flashcard
greedily apply learned merges to new text

**Special tokens?** #flashcard
<|endoftext|> marks document boundaries; <|unk|> rarely needed with BPE

**Token embedding?** #flashcard
map each integer token ID to a d-dimensional vector via a lookup table (trained)

**Simplified self-attention (no parameters)?** #flashcard
A = softmax(X Xᵀ / √d) · X

**Trainable self-attention?** #flashcard
Q=XW_Q, K=XW_K, V=XW_V; A = softmax(QKᵀ/√d_k)·V

**Causal mask?** #flashcard
set upper-triangle of QKᵀ to -inf before softmax

**Multi-head?** #flashcard
run h heads with d_k = d_model/h; concatenate outputs; project with W_O

**Cosine LR schedule with warmup?** #flashcard
linearly increase to peak LR over first 2% of steps, then cosine decay

**AdamW optimizer?** #flashcard
Adam + weight decay on all non-bias/non-norm parameters

**Evaluate perplexity on held-out validation text every N steps?** #flashcard
Evaluate perplexity on held-out validation text every N steps

**Save checkpoints; load GPT-2 weights with model.load_state_dict(torch.load(...))?** #flashcard
Save checkpoints; load GPT-2 weights with model.load_state_dict(torch.load(...))

**Remove LM head; add nn.Linear(d_model, num_classes)?** #flashcard
Remove LM head; add nn.Linear(d_model, num_classes)

**Forward pass?** #flashcard
take hidden state at last token position → classification logits

**Loss?** #flashcard
cross-entropy on class labels

**Selective freezing?** #flashcard
freeze all transformer blocks, train only the head (faster, less data needed, more risk of underfitting)

**Full fine-tuning?** #flashcard
unfreeze all parameters (slower, needs more data, usually better)

**Format?** #flashcard
<|system|>You are a helpful assistant.<|user|>{instruction}<|assistant|>{response}

**Supervised fine-tuning (SFT)?** #flashcard
compute loss only on the response tokens (mask instruction tokens)

**RLHF pipeline:?** #flashcard
RLHF pipeline:

**LoRA for efficient fine-tuning?** #flashcard
freeze base model; add low-rank matrices ΔW = A·B to attention projections; train only A and B

**nn.Module?** #flashcard
base class for all models; .parameters() yields all trainable tensors; .train()/.eval() switches dropout/BN behavior

**DataLoader?** #flashcard
wraps a Dataset; handles batching, shuffling, multi-process loading

**optimizer.zero_grad() → loss.backward() → optimizer.step()?** #flashcard
standard training loop

**torch.no_grad()?** #flashcard
context manager that disables gradient tracking (inference, evaluation)

**Model loading?** #flashcard
model.load_state_dict(state_dict, strict=False) allows partial loading (e.g., loading GPT-2 weights into custom architecture)
