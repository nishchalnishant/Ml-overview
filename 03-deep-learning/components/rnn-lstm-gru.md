# RNN, LSTM, and GRU

---

## TL;DR

Vanilla RNNs process sequences step-by-step with a shared hidden state, but gradients vanish or explode through long sequences. LSTMs fix this with a gated cell state that acts as a direct gradient highway. GRUs simplify LSTMs to two gates with comparable performance. Both are largely superseded by Transformers for parallelizable tasks, but remain relevant for streaming, on-device inference, and as the conceptual foundation for SSMs (Mamba).

---

## Vanilla RNN

**The problem**: process a variable-length sequence while maintaining context. A feedforward network sees one timestep at a time and has no memory.

**The core insight**: at each timestep, mix the current input with the previous hidden state using shared weights. The hidden state carries a compressed summary of all previous inputs.

**The mechanics**:

$$h_t = \tanh(W_h h_{t-1} + W_x x_t + b)$$

$$\hat{y}_t = W_y h_t + b_y$$

- $h_t \in \mathbb{R}^{d_h}$: hidden state at time $t$
- $x_t \in \mathbb{R}^{d_x}$: input at time $t$
- $W_h \in \mathbb{R}^{d_h \times d_h}$, $W_x \in \mathbb{R}^{d_h \times d_x}$: shared across all timesteps

**Unrolling**: mentally copy the RNN cell once per timestep and stack them left to right. Each copy shares the same weights. Backpropagation traverses this unrolled graph back through all timesteps — hence BPTT.

**Parameter count**: $d_h \times d_h + d_h \times d_x + d_h$ (bias). For $d_h = 256$, $d_x = 128$: $256^2 + 256 \times 128 + 256 = 98,560$. Weight sharing is the key: the same $W_h$ is reused at every timestep regardless of sequence length.

**What breaks**: gradients either vanish or explode as they propagate backward through many timesteps. Useful memory effectively spans only ~10–20 steps for a standard RNN.

---

## Backpropagation Through Time (BPTT)

**The problem**: computing gradients in an unrolled RNN requires propagating errors back through every timestep — a chain of multiplications across potentially hundreds of steps.

**The mechanics**: the loss at the final step $T$ depends on $h_T$, which depends on $h_{T-1}$, ..., which depends on $h_1$. By the chain rule:

$$\frac{\partial L}{\partial h_t} = \frac{\partial L}{\partial h_T} \prod_{k=t+1}^{T} \frac{\partial h_k}{\partial h_{k-1}}$$

Each Jacobian factor is:

$$\frac{\partial h_k}{\partial h_{k-1}} = \text{diag}(\tanh'(z_k)) \cdot W_h$$

where $z_k = W_h h_{k-1} + W_x x_k + b$.

**The gradient product**: the gradient from timestep $T$ back to timestep $t$ involves the product of $(T - t)$ matrices. For sequences of length 100, this is a product of 100 matrices — which either collapses to zero or diverges.

**Truncated BPTT**: in practice, gradients are only propagated back $k$ steps (typically 20–50). Hidden states are carried forward without gradients beyond the truncation window. This is the default in most RNN training code.

---

## Vanishing and Exploding Gradients

**The mathematical cause**: consider the simplified case where $\tanh' \approx 1$ (near-linear regime). The gradient product becomes $W_h^{T-t}$. The behavior is governed by the eigenvalues of $W_h$:

- If $|\lambda_{\max}| < 1$: gradients shrink geometrically — **vanishing gradients**. Early timesteps receive near-zero gradient signal, so the model cannot learn long-range dependencies.
- If $|\lambda_{\max}| > 1$: gradients grow geometrically — **exploding gradients**. Training becomes numerically unstable, with loss spiking to NaN.

**Why vanishing is harder to fix than exploding**:

| Problem | Effect | Detection | Fix |
|---|---|---|---|
| Exploding | Loss goes to NaN, huge weight updates | Easy — loss diverges | Gradient clipping |
| Vanishing | Model ignores distant context | Subtle — model just performs poorly | Architecture change (LSTM/GRU) |

**Gradient clipping** (exploding gradients fix):

$$\text{if } \|\nabla\|_2 > \tau, \quad \nabla \leftarrow \frac{\tau}{\|\nabla\|_2} \nabla$$

Clips the global gradient norm, not individual parameter gradients. Standard threshold: $\tau = 1.0$ to $5.0$.

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**Orthogonal initialization** (vanishing gradients mitigation): initialize $W_h$ as an orthogonal matrix. Orthogonal matrices have all eigenvalues with $|\lambda| = 1$, so the product $W_h^T$ neither grows nor shrinks. Helps at initialization; does not prevent drift during training.

```python
torch.nn.init.orthogonal_(rnn.weight_hh_l0)
```

---

## LSTM (Long Short-Term Memory)

**The core insight**: introduce a second state — the **cell state** $c_t$ — that flows through time with only additive updates (not multiplicative). Gates are learned sigmoid functions that control what information is written, erased, or read.

**The four gates**:

$$f_t = \sigma(W_f [h_{t-1}, x_t] + b_f) \quad \text{(forget gate)}$$

$$i_t = \sigma(W_i [h_{t-1}, x_t] + b_i) \quad \text{(input gate)}$$

$$\tilde{c}_t = \tanh(W_c [h_{t-1}, x_t] + b_c) \quad \text{(cell candidate)}$$

$$o_t = \sigma(W_o [h_{t-1}, x_t] + b_o) \quad \text{(output gate)}$$

**Cell and hidden state updates**:

$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$

$$h_t = o_t \odot \tanh(c_t)$$

where $\odot$ is element-wise multiplication, $\sigma$ is sigmoid.

**Gate intuitions**:

| Gate | Range | Intuition |
|---|---|---|
| Forget $f_t$ | $[0, 1]$ | How much of the old cell state to keep. 0 = erase, 1 = keep. |
| Input $i_t$ | $[0, 1]$ | How much of the new candidate to write. |
| Cell candidate $\tilde{c}_t$ | $[-1, 1]$ | What new information to potentially add. |
| Output $o_t$ | $[0, 1]$ | How much of the cell to expose as the hidden state. |

**Why cell state solves vanishing gradients (Constant Error Carousel)**:

The gradient of the cell state is:

$$\frac{\partial c_t}{\partial c_{t-1}} = f_t$$

This is a simple element-wise multiplication — no matrix multiply, no $\tanh'$ squashing. If the forget gate stays near 1 (which it learns to do when the task requires long memory), gradient flows backward through $c_t$ without attenuation. This is the "constant error carousel" from the original Hochreiter & Schmidhuber (1997) paper.

**Parameter count**: four weight matrices each of shape $(d_h, d_h + d_x)$ plus biases. Total: $4 \times (d_h(d_h + d_x) + d_h) = 4d_h(d_h + d_x + 1)$.

For $d_h = 256$, $d_x = 128$: $4 \times 256 \times 385 = 394,240$. Roughly 4x a vanilla RNN.

**Forget gate bias initialization**: initialize $b_f$ to 1.0 (or 2.0) so the forget gate starts near 1. This means the cell initially remembers everything, and the model learns what to forget rather than starting from amnesia.

---

## GRU (Gated Recurrent Unit)

**The core insight**: simplify LSTM by merging cell state and hidden state, and combining the forget and input gates into a single update gate. Fewer parameters, often comparable performance.

**The two gates**:

$$r_t = \sigma(W_r [h_{t-1}, x_t] + b_r) \quad \text{(reset gate)}$$

$$z_t = \sigma(W_z [h_{t-1}, x_t] + b_z) \quad \text{(update gate)}$$

**Hidden state update**:

$$\tilde{h}_t = \tanh(W_h [r_t \odot h_{t-1}, x_t] + b_h) \quad \text{(candidate hidden state)}$$

$$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$

**Gate intuitions**:

| Gate | Intuition |
|---|---|
| Reset $r_t$ | How much of the previous hidden state to use when computing the candidate. Near 0 = ignore past (start fresh). |
| Update $z_t$ | How much to blend old hidden state vs. new candidate. Near 0 = keep old, near 1 = use new. |

**GRU vs LSTM**:

| | LSTM | GRU |
|---|---|---|
| Gates | 4 (forget, input, cell update, output) | 2 (reset, update) |
| States | $h_t$ and $c_t$ | $h_t$ only |
| Parameters | $4d_h(d_h + d_x + 1)$ | $3d_h(d_h + d_x + 1)$ |
| Gradient highway | Cell state $c_t$ | Blended $h_t$ via update gate |
| Performance | Slightly better on some long-range tasks | Comparable, faster to train |
| When to prefer | Tasks needing fine-grained control of memory | Default choice when experimenting |

---

## Bidirectional RNNs

**The problem**: a standard RNN at position $t$ can only see tokens $1, \ldots, t$. For tasks like NER or sentiment classification, the word at position $t$ often depends on future context.

**The core insight**: run two RNNs — one forward, one backward — and concatenate their hidden states at each position.

$$\overrightarrow{h}_t = \text{RNN}(x_t, \overrightarrow{h}_{t-1}) \quad \text{(forward)}$$

$$\overleftarrow{h}_t = \text{RNN}(x_t, \overleftarrow{h}_{t+1}) \quad \text{(backward)}$$

$$h_t = [\overrightarrow{h}_t; \overleftarrow{h}_t] \in \mathbb{R}^{2d_h}$$

**When it helps**:
- Sequence labeling (NER, POS tagging) — each label benefits from full context
- Sentence classification — the final representation captures both directions
- Machine translation encoder — the encoder can be bidirectional; only the decoder must be causal

**Inference limitation**: the backward pass requires the full sequence before it can start. This makes bidirectional RNNs **non-autoregressive** — you cannot generate tokens one at a time. They are encoders, not decoders.

**In PyTorch**:

```python
rnn = nn.LSTM(input_size=128, hidden_size=256, bidirectional=True)
# output shape: (seq_len, batch, 512)  # 2 * hidden_size
```

---

## Stacked / Deep RNNs

**The motivation**: a single RNN layer learns one level of temporal abstraction. Stacking allows hierarchical representations — lower layers capture local patterns, higher layers capture longer-range structure.

$$h_t^{(l)} = \text{RNN}(h_t^{(l-1)}, h_{t-1}^{(l)})$$

The output of layer $l-1$ at time $t$ serves as the input to layer $l$ at time $t$.

**Depth in practice**: 2–4 layers is typical. More than 4 layers rarely helps and increases training instability. ELMo used 2-layer BiLSTMs. Encoder-decoder seq2seq models commonly used 4-layer stacked LSTMs.

**Dropout between layers**: apply dropout to the connections between layers (not the recurrent connections, which would destroy temporal information).

```python
nn.LSTM(input_size=128, hidden_size=256, num_layers=4, dropout=0.3)
# dropout applies between layers 1-2, 2-3, 3-4 (not after the last layer)
```

**Recurrent dropout** (Gal & Ghahramani 2016): sample one dropout mask per sequence (not per timestep) and apply the same mask at every step. This preserves gradient flow in the time direction while still regularizing.

---

## RNN vs Transformer

| Dimension | RNN / LSTM | Transformer |
|---|---|---|
| **Parallelism** | Sequential — step $t$ requires step $t-1$. Cannot parallelize across time. | Fully parallel across all positions during training. |
| **Long-range dependencies** | Degrade exponentially. LSTM extends range but still struggles at >100 steps. | $O(1)$ path length between any two positions via attention. |
| **Memory** | $O(1)$ state at inference (just $h_t$). Constant memory regardless of sequence length. | $O(n)$ KV cache at inference grows with context. |
| **Training compute** | $O(n \cdot d_h^2)$ per sequence. | $O(n^2 \cdot d)$ — quadratic in sequence length. |
| **Inductive bias** | Strong positional bias — locality matters by construction. | No positional bias; requires explicit positional encoding. |
| **Variable-length sequences** | Natural fit — just run until end token. | Requires padding/masking; packing reduces waste. |
| **Autoregressive generation** | Efficient — one step at a time, constant compute per step. | KV cache amortizes cost but memory grows. |

**When RNNs still win**:
- Online / streaming inference where input arrives one token at a time and latency matters
- Extremely long sequences where $O(n^2)$ attention is prohibitive and the full context isn't needed
- Memory-constrained deployment (edge/embedded devices)
- Time series with strong local temporal structure

---

## Modern Successors

**Why Transformers replaced RNNs**: the parallelism advantage is decisive during training. On modern GPU/TPU hardware, sequential operations destroy utilization. A Transformer training on a 512-token sequence is ~512x more parallelizable than an LSTM on the same sequence.

**The remaining gap**: Transformers are quadratic in sequence length. For contexts of 100K+ tokens, attention becomes the bottleneck.

**SSMs (State Space Models) — Mamba as the third path**:

SSMs model sequences via a latent state $h_t$ governed by learned continuous-time dynamics:

$$h_t = \bar{A} h_{t-1} + \bar{B} x_t$$
$$y_t = C h_t$$

- Can be computed as a convolution during training (parallel like Transformers)
- Can be computed as a recurrence during inference (constant memory like RNNs)
- Linear in sequence length: $O(n \cdot d)$

**Mamba** (Gu & Dao, 2023) makes $\bar{A}$, $\bar{B}$, $C$ input-dependent (selective), which recovers the expressiveness that fixed SSMs lack. Achieves competitive performance with Transformers on language modeling at significantly reduced memory cost for long sequences.

The current landscape: Transformers dominate most tasks; Mamba/SSMs are promising for long-context and streaming applications; pure LSTMs are legacy for most NLP tasks but still used in production systems with strict latency/memory budgets.

---

## Practical Considerations

**Weight initialization**:
- $W_x$: Xavier/Glorot initialization (`nn.init.xavier_uniform_`)
- $W_h$: Orthogonal initialization to avoid eigenvalue scaling issues at the start
- Forget gate bias: initialize to 1.0 in LSTMs (`b_f = 1.0`)
- Output projection: small normal initialization

**Gradient clipping implementation**:

```python
# Clip before optimizer step
optimizer.zero_grad()
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

Global norm clipping (above) is preferred over per-parameter clipping — it preserves gradient direction while limiting magnitude.

**Sequence packing for variable-length inputs**: naively, batching sequences of different lengths requires padding shorter sequences, then masking the padded positions. This wastes compute on padding tokens.

PyTorch's `pack_padded_sequence` and `pad_packed_sequence` pack sequences densely:

```python
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# Sort by length descending (required for PackedSequence)
packed = pack_padded_sequence(padded_input, lengths, batch_first=True, enforce_sorted=False)
output_packed, (h_n, c_n) = lstm(packed)
output, _ = pad_packed_sequence(output_packed, batch_first=True)
```

The RNN only processes actual tokens; no wasted compute on padding positions.

**Hidden state initialization**: default to zeros. At evaluation time, optionally carry the final hidden state across chunks of a long sequence (stateful inference).

---

## Interview Questions

**Q1: Why do vanilla RNNs fail on long sequences, and how does the LSTM cell state fix this?**

Vanilla RNN gradients involve products of Jacobians $\prod \text{diag}(\tanh') \cdot W_h$. If the dominant eigenvalue of $W_h$ is less than 1, this product vanishes exponentially. The LSTM cell state update $c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$ has a gradient $\partial c_t / \partial c_{t-1} = f_t$ — element-wise, no matrix multiply. When $f_t \approx 1$, the gradient flows unattenuated. This is the constant error carousel.

---

**Q2: What is the difference between gradient clipping by value vs. by norm? Which is preferred?**

Clipping by value caps each gradient element independently: $g \leftarrow \text{clip}(g, -\tau, \tau)$. This distorts the gradient direction. Clipping by global norm scales the entire gradient vector uniformly if its norm exceeds $\tau$: $g \leftarrow g \cdot \tau / \|g\|$. Global norm clipping preserves gradient direction and is preferred. The heuristic is: if you're clipping frequently, your learning rate may also be too high.

---

**Q3: When would you choose a GRU over an LSTM?**

GRU has 3/4 the parameters of LSTM and trains faster. On most tasks the gap is small. Prefer GRU when: compute budget is tight, sequences are moderately long (not extremely long-range dependencies), or as a default when doing architecture search. Prefer LSTM when: task requires nuanced memory management (e.g., certain language modeling tasks), or you have evidence from experiments that it outperforms GRU.

---

**Q4: Why can't you use a bidirectional RNN as a language model (autoregressive decoder)?**

Autoregressive generation requires predicting token $t$ from tokens $1, \ldots, t-1$ only. A bidirectional RNN's representation at position $t$ incorporates tokens $t+1, \ldots, T$ — it "sees the future." This is valid for encoders (sentence classification, NER) where the full input is available, but invalid for decoders where future tokens don't exist yet at generation time.

---

**Q5: Explain truncated BPTT. What are its trade-offs?**

Truncated BPTT propagates gradients only $k$ steps back (e.g., $k = 35$), then stops. Hidden states are passed forward without carrying gradients. Trade-off: (a) the model cannot learn dependencies longer than $k$ steps; (b) gradient estimates are biased because the recurrent weight update ignores how $h_t$ affects future losses beyond the window; (c) in practice this is acceptable because most relevant context is local anyway, and it makes training tractable for long sequences.

---

**Q6: How does Mamba (SSM) achieve both parallel training and efficient autoregressive inference, when RNNs cannot?**

SSMs have a dual form: as a recurrence during inference ($h_t = \bar{A} h_{t-1} + \bar{B} x_t$, one step at a time) and as a convolution during training ($y = x * \bar{K}$ where $\bar{K}$ is the SSM kernel, computed in $O(n \log n)$ via FFT). RNNs lack a convolutional form because their recurrence involves nonlinearities ($\tanh$, sigmoid) that prevent exact convolution. Mamba makes the SSM matrices input-dependent (selective), adding expressiveness while maintaining the convolutional/recurrent duality.

---

**Q7: What happens if you initialize the hidden state of an LSTM with the final hidden state from the previous batch (stateful LSTM)? When is this useful and what are the pitfalls?**

Useful for: processing very long sequences (e.g., audio, long documents) that are chunked into fixed-length windows. The model sees a continuous stream rather than independent chunks. Pitfalls: (a) batch order must be consistent — all sequences in a batch must have their chunks presented in the right order; (b) gradient flow is still truncated at chunk boundaries unless you also carry the computation graph (expensive); (c) at evaluation, you must reset state at sequence boundaries carefully to avoid contamination between independent sequences.
