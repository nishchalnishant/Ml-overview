# Transformers

Transformers are encoder–decoder (or decoder-only) architectures built on **self-attention** and **feed-forward layers**, without recurrent connections. They underlie modern LLMs and most of NLP and multimodal AI.

---

## Architecture overview

- **Encoder**: stack of layers, each with multi-head self-attention + feed-forward; residual connections and layer norm.
- **Decoder**: same, plus **causal self-attention** (each position sees only past) and **cross-attention** to encoder output.
- **Decoder-only** (GPT-style): no encoder; only causal self-attention + feed-forward. Used for autoregressive language modeling and most LLMs.

---

## Single transformer layer (decoder block)

1. **Causal multi-head self-attention**  
   - Q, K, V from same sequence; mask so position \(t\) sees only positions \(\le t\).  
   - Output: residual + LayerNorm.

2. **Cross-attention** (encoder–decoder only)  
   - Q from decoder, K and V from encoder.  
   - Residual + LayerNorm.

3. **Feed-forward (FFN)**  
   - Two linear layers with activation (e.g. GELU) in between: \(\text{FFN}(x) = W_2\,\sigma(W_1 x)\).  
   - Residual + LayerNorm.

---

## Key formulas

**Self-attention (single head):**
\[
\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
\]

**Positional encoding** (original transformer):  
Sinusoidal functions so the model knows token order:
\[
PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d}), \quad PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d})
\]
Many modern models use **learned** positional embeddings instead.

---

## Encoder vs decoder vs decoder-only

| Model type | Encoder | Decoder | Typical use |
|------------|--------|---------|-------------|
| **Encoder-only** (BERT) | ✓ (bidirectional) | — | Classification, NER, embeddings |
| **Decoder-only** (GPT, LLaMA) | — | ✓ (causal only) | Text generation, chat, most LLMs |
| **Encoder–decoder** (T5, BART) | ✓ | ✓ + cross-attention | Translation, summarization, Q&A over context |

---

## Pretraining vs fine-tuning

- **Pretraining**: train on large unlabeled/semi-labeled text (next-token prediction for decoder-only; masked LM or similar for encoder).
- **Fine-tuning**: adapt on task-specific data (classification, QA, etc.).
- **Instruction tuning**: fine-tune on (instruction, response) pairs so the model follows user intents.
- **RLHF / DPO**: align outputs with human preferences (see [RLHF and alignment](#connection-to-rlhf-and-alignment) below).

---

## Connection to modern AI systems

- **LLMs**: GPT, LLaMA, Mistral, etc. are decoder-only transformers at scale.
- **RAG**: retrieval feeds context into the transformer’s context window; model attends over it via attention.
- **Agents**: same LLM backbone; tool use and planning are orchestrated around the transformer’s I/O (prompts, tool calls, observations).

---

## Connection to RLHF and alignment

- **RLHF (Reinforcement Learning from Human Feedback)**: reward model trained on human preferences; policy (LLM) optimized with RL (e.g. PPO) to maximize reward.
- **DPO (Direct Preference Optimization)**: align with preferences without a separate reward model or RL loop.
- **Instruction tuning** often precedes RLHF/DPO so the model first learns to follow instructions.

---

## Quick revision

- **Transformer** = stacked layers of (multi-head) self-attention + FFN, with residuals and LayerNorm.
- **Causal mask** in decoder ensures autoregressive generation; **cross-attention** in encoder–decoder links decoder to encoder.
- **Decoder-only** = no encoder; used for most LLMs. **Encoder–decoder** used for conditional generation (translation, summarization).
- **Pretraining** → **fine-tuning** → **instruction tuning** → **RLHF/DPO** is the typical path to aligned, instruction-following models.
