---
module: Deep Learning
topic: Revision Card
subtopic: ""
status: unread
tags: [deep-learning, revision, cheatsheet]
---
# Deep Learning — 10-Minute Revision Card

---

## Architecture Decision Tree

| Task | Start with |
|------|-----------|
| Images (classification) | ResNet / EfficientNet |
| Images (detection) | YOLO / Faster R-CNN |
| Sequence (generation) | Transformer decoder |
| Sequence (understanding) | BERT-style encoder |
| Tabular | Gradient Boosting first, then MLP |
| Time series | Temporal Conv / Transformer |
| Small data + images | Pretrained CNN + fine-tune top layers |

---

## Backpropagation in 60 Seconds

**Problem:** assign credit to millions of weights from a single scalar loss.

**Chain rule:** $\frac{\partial L}{\partial w} = \frac{\partial L}{\partial a} \cdot \frac{\partial a}{\partial z} \cdot \frac{\partial z}{\partial w}$

**Vanishing gradient:** sigmoid derivative ≤ 0.25 → $0.25^{10} \approx 10^{-6}$ after 10 layers → early layers stop learning.

**Fixes (in order of effectiveness):**
1. ReLU activations (gradient = 1 for positive inputs)
2. Residual connections (gradient highway: $x_{l+1} = x_l + F(x_l)$)
3. Layer/Batch Normalization (prevents saturation)
4. He initialization (keeps signal variance stable across layers)

**Exploding gradients:** opposite problem — clip gradients (`max_norm=1.0`).

---

## Activation Functions

| Activation | Formula | Use | Gotcha |
|-----------|---------|-----|--------|
| ReLU | $\max(0, x)$ | Default hidden layers | Dead neurons (neg input → zero gradient always) |
| LeakyReLU | $\max(0.01x, x)$ | When dying ReLU is a problem | Small negative slope (0.01) allows gradient flow |
| GELU | $x \cdot \Phi(x)$ | Transformers | Smooth approximation, better than ReLU in practice |
| Sigmoid | $\frac{1}{1+e^{-z}}$ | Binary output only | Saturates → vanishing gradient in hidden layers |
| Softmax | $\frac{e^{z_i}}{\sum e^{z_j}}$ | Multiclass output | Numerically unstable without log-sum-exp trick |

---

## Optimizers

| Optimizer | Key insight | Use |
|-----------|------------|-----|
| SGD + momentum | Gradient + velocity term; escapes local minima | Vision models with LR schedule |
| Adam | Adaptive per-parameter LR; momentum + RMS scaling | Default for NLP/transformers |
| AdamW | Adam + decoupled weight decay | Transformers (L2 reg done right) |

**Adam formulas:**
- $m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t$ (momentum)
- $v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2$ (variance)
- $\theta_t = \theta_{t-1} - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$

**Gotcha:** Adam's adaptive scaling interacts with L2 regularization — always use AdamW for transformers, not Adam + weight_decay.

---

## Attention Mechanism

**Problem:** RNNs compress entire sequence into fixed-size state → information loss at long range.

**Solution:** let every token attend directly to every other token.

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

- **Q (Query):** what this token is looking for
- **K (Key):** what each token offers
- **V (Value):** what each token returns
- **$\sqrt{d_k}$ scaling:** prevents dot products from growing large → softmax saturation → near-zero gradients

**Multi-head attention:** H independent attention heads with different projections → each specializes (syntax, coreference, semantics, etc.)

**Self vs Cross:**
- Self-attention: Q, K, V from same sequence → contextual understanding
- Cross-attention: Q from one sequence, K/V from another → RAG, encoder-decoder

**Causal masking:** mask future positions to $-\infty$ → softmax gives zero weight → autoregressive generation works correctly. Makes model unidirectional — wrong for BERT-style understanding.

**Complexity:** $O(T^2 \cdot d)$ — quadratic in sequence length. At T=8192, this is the bottleneck.

---

## Transformers

**Encoder block:** Multi-head self-attention → Add & Norm → FFN → Add & Norm

**Decoder block:** Masked self-attention → Cross-attention (to encoder) → FFN → each with Add & Norm

**Positional encoding:** attention has no sense of order; add sinusoidal or learned position embeddings.

**FFN is 2/3 of parameters:** $d_{model} \rightarrow 4 \times d_{model} \rightarrow d_{model}$ — most knowledge stored here, not in attention.

**Layer Norm placement:**
- Post-LN (original paper): normalize after residual → training instability at scale
- Pre-LN (modern): normalize before sublayer → stable training, most models use this

---

## Regularization in Deep Learning

| Technique | How | When |
|-----------|-----|------|
| Dropout | Zero activations randomly during training | FC layers; not attention (usually) |
| Weight Decay (L2) | Add $\lambda \|w\|^2$ to loss | Almost always |
| Batch Norm | Normalize activations per mini-batch | CNNs; problematic with small batch |
| Layer Norm | Normalize per token/example | Transformers |
| Early Stopping | Stop when val loss stagnates | Always use |
| Data Augmentation | Synthetic training examples | Vision; some NLP |

---

## Training Diagnostics

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| Loss NaN immediately | LR too high or bad init | Lower LR, check init |
| Train loss low, val loss high | Overfitting | More data, dropout, weight decay |
| Both losses high and flat | Vanishing gradient or LR too low | Check activations, raise LR |
| Loss oscillates wildly | LR too high | LR schedule, gradient clipping |
| Slow convergence | LR too low or bad batch size | Increase LR, try warmup |
| GPU memory OOM | Batch too large | Gradient accumulation, reduce batch |

---

## Transfer Learning Decision Tree

1. **Data < 1k examples:** freeze all layers, train only classifier head
2. **Data 1k–10k:** freeze early layers, fine-tune last 2–3 blocks
3. **Data > 10k:** fine-tune entire network with lower LR for pretrained layers
4. **Domain very different from ImageNet/NLP pretraining:** full fine-tune or train from scratch

**Why it works:** early layers learn universal features (edges, curves, tokens) → later layers learn task-specific patterns. Fine-tuning adapts the latter while retaining the former.

---

## CNN Building Blocks

- **Conv layer:** kernel slides over input, learns local patterns. Output size: $\lfloor\frac{W - K + 2P}{S}\rfloor + 1$
- **Pooling:** reduce spatial dimensions. MaxPool keeps sharpest feature; AvgPool smooths.
- **Receptive field:** grows with depth — deeper layers "see" larger input regions.
- **Depthwise separable conv (MobileNet):** factorize $K^2 \times C_{in} \times C_{out}$ into depthwise ($K^2 \times C_{in}$) + pointwise ($C_{in} \times C_{out}$) → ~8–9× fewer operations.

---

## PyTorch Pattern Cheatsheet

```python
# Training loop skeleton
model.train()
optimizer.zero_grad()
output = model(x)
loss = criterion(output, y)
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
scheduler.step()

# Validation
model.eval()
with torch.no_grad():
    output = model(x)
```

**Common mistakes:**
- Forgetting `optimizer.zero_grad()` → gradients accumulate
- Running validation without `torch.no_grad()` → unnecessary memory
- Calling `loss.backward()` on non-scalar → need `.mean()` or `.sum()` first

---

## Interview Quick-Draws

**"Why does batch normalization help?"**
→ Reduces internal covariate shift — each layer sees inputs with stable distribution. Also acts as regularizer (noise from batch statistics). Allows higher LRs.

**"What's the difference between Layer Norm and Batch Norm?"**
→ BN normalizes across batch dimension (needs large batch, breaks at small batch). LN normalizes across feature dimension per example → works for any batch size → preferred in transformers.

**"Explain attention in one minute."**
→ Each token computes a weighted sum over all tokens' values, where weights come from query-key similarity. Lets model route information selectively across arbitrary distances. Complexity is O(T²).

**"Why residual connections?"**
→ Gradient highway: addition operation has gradient = 1 so gradient flows unchanged through skip path. Enables training of very deep networks. Also provides identity path (model can learn no-op).

**"Dropout at inference time?"**
→ Disabled at inference. Scale activations by keep probability (1-p) at training, or equivalently scale outputs by keep probability at test time (inverted dropout — what PyTorch does).
