# Regularization

Regularization is how you stop a deep model from becoming wildly overconfident about noisy patterns.

It is not about making the model weak. It is about making it less gullible.

---

# 1. L1 and L2 Regularization

Add a penalty term to the loss function:

**L2 (Ridge / Weight Decay):**

$$L_{\text{total}} = L_{\text{task}} + \frac{\lambda}{2} \sum_w w^2$$

Gradient penalty: $\frac{\partial}{\partial w} = \lambda w$ — shrinks all weights proportionally.

Effect: smaller but non-zero weights. Smooth, stable. Default choice in deep learning.

**L1 (Lasso):**

$$L_{\text{total}} = L_{\text{task}} + \lambda \sum_w |w|$$

Gradient penalty: $\lambda \cdot \text{sign}(w)$ — constant push toward zero.

Effect: encourages **sparsity** — many weights become exactly zero. Useful for feature selection in linear models.

**Elastic Net:** combines L1 and L2:

$$L_{\text{total}} = L_{\text{task}} + \lambda_1 \sum |w| + \frac{\lambda_2}{2} \sum w^2$$

```python
# L2 weight decay is built into most optimizers
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)

# Manual L2 addition (if using Adam without AdamW):
l2_penalty = sum(p.pow(2).sum() for p in model.parameters())
loss = task_loss + 1e-4 * l2_penalty
```

---

# 2. Dropout

Randomly zero activations during training with probability $p$:

$$h' = \frac{h \cdot m}{1-p}, \quad m \sim \text{Bernoulli}(1-p)$$

The $\frac{1}{1-p}$ scaling (**inverted dropout**) ensures the expected value of $h'$ equals $h$, so no scaling is needed at test time.

**Why it works:**
- Forces the network to learn redundant representations
- Prevents co-adaptation of neurons (no single neuron can be relied on)
- Ensemble interpretation: training with dropout ≈ averaging over $2^n$ sub-networks

**Typical rates:**
- Dense layers: $p = 0.5$
- Smaller layers / CNNs: $p = 0.1$–$0.3$
- Transformers: $p = 0.1$

**Inference:** disable dropout (`model.eval()` in PyTorch handles this automatically).

```python
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Dropout(p=0.5),    # applied only during training
    nn.Linear(256, 10)
)

# model.train() enables dropout
# model.eval() disables dropout — critical for correct inference
```

---

# 3. BatchNorm and Regularization Effect

BatchNorm is primarily an **optimization technique** (smoother loss landscape, allows higher learning rates) but has a mild regularizing effect:

- Adds noise via mini-batch statistics during training (acts like data augmentation)
- Reduces sensitivity to initialization and learning rate

Layer formulation:

$$\hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}, \quad y = \gamma \hat{x} + \beta$$

$\mu_B$, $\sigma_B^2$: mini-batch mean and variance. $\gamma$, $\beta$: learned scale and shift.

**At inference:** uses running mean/variance accumulated during training (not batch stats).

When BatchNorm is used heavily, you can often reduce dropout rate.

---

# 4. Layer Normalization

LayerNorm normalizes across features (not across the batch):

$$\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}, \quad \text{where } \mu, \sigma \text{ computed over feature dimension}$$

**Used in:** Transformers (batch size of 1 is fine; no dependence on batch statistics).

**Pre-norm vs Post-norm:**
- **Post-norm** (original Transformer): `LayerNorm(x + Sublayer(x))` — harder to train deep networks
- **Pre-norm** (modern default): `x + Sublayer(LayerNorm(x))` — more stable, scales better

---

# 5. Early Stopping

Train until validation metric stops improving; save the checkpoint at the best validation point.

Very practical and underrated — prevents overfitting without modifying the model architecture.

```python
best_val_loss = float('inf')
patience, patience_counter = 10, 0

for epoch in range(max_epochs):
    train_one_epoch(model, optimizer)
    val_loss = evaluate(model, val_loader)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pt')
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

model.load_state_dict(torch.load('best_model.pt'))
```

---

# 6. Data Augmentation

Especially important in vision. Augmentation exposes the model to broader variation and teaches useful invariances.

**Standard vision augmentations:**
```python
import torchvision.transforms as T

train_transform = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomCrop(32, padding=4),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    T.RandomRotation(15),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

**MixUp:** linearly interpolate between pairs of training examples and their labels:

$$\tilde{x} = \lambda x_i + (1-\lambda) x_j, \quad \tilde{y} = \lambda y_i + (1-\lambda) y_j, \quad \lambda \sim \text{Beta}(\alpha, \alpha)$$

**α guidance:**
- $\alpha = 0.2$: mild mixing, $\lambda$ usually close to 0 or 1 → near-clean examples. Good default.
- $\alpha = 1.0$: uniform distribution over $[0,1]$ → more aggressive mixing.
- $\alpha > 1.0$: $\lambda$ concentrated near 0.5 → very blended examples, can hurt single-label tasks.
- Start with $\alpha \in [0.2, 0.4]$ and tune based on validation performance.

**CutMix:** paste a random patch from one image into another; mix labels proportionally to patch area.

Both MixUp and CutMix act as strong regularizers and improve calibration.

---

# 7. DropConnect

Generalization of Dropout: instead of zeroing activations, randomly zero individual **weights** during training:

$$h = (W \odot M) x, \quad M_{ij} \sim \text{Bernoulli}(1-p)$$

- Dropout masks neurons; DropConnect masks connections
- More fine-grained stochasticity but computationally heavier
- Less commonly used in practice; Dropout is usually sufficient

---

# 7a. Spectral Normalization

Constrain the spectral norm (largest singular value) of each weight matrix to 1:

$$\hat{W} = \frac{W}{\sigma(W)}, \quad \sigma(W) = \text{largest singular value of } W$$

Approximated efficiently using power iteration:

$$\tilde{v} = W^T \hat{u}, \quad \hat{v} = \tilde{v}/\|\tilde{v}\|_2$$
$$\tilde{u} = W \hat{v}, \quad \hat{u} = \tilde{u}/\|\tilde{u}\|_2$$
$$\sigma(W) \approx \hat{u}^T W \hat{v}$$

Effect: Lipschitz constraint on each layer → stable discriminator training in GANs, more stable training generally.

```python
import torch.nn as nn

# Apply spectral norm to a layer
layer = nn.utils.spectral_norm(nn.Linear(256, 256))
```

Used in: GANs (SNGAN), self-supervised learning, any setting where Lipschitz continuity matters.

---

# 8. DropPath / Stochastic Depth

Randomly drop entire residual branches during training (not individual neurons):

$$x_{l+1} = x_l + b_l \cdot F_l(x_l), \quad b_l \sim \text{Bernoulli}(p_l)$$

Drop probability $p_l$ increases linearly with layer depth. Used in: ViT, EfficientNet, Swin Transformer.

Effect: shortens the effective network depth stochastically → reduces gradient vanishing, regularizes.

---

# 8. Regularization Summary

| Technique | Mechanism | Best For |
| :--- | :--- | :--- |
| **L2 weight decay** | Penalizes large weights | All networks (default) |
| **L1** | Encourages sparsity | Linear models, feature selection |
| **Dropout** | Random neuron masking | Dense layers, Transformers |
| **BatchNorm** | Batch normalization | CNNs, feedforward nets |
| **LayerNorm** | Feature normalization | Transformers (required) |
| **Early stopping** | Stop at best val checkpoint | Universal |
| **Data augmentation** | Expose model to variation | Vision, audio |
| **MixUp/CutMix** | Label-preserving interpolation | Vision, strong regularizer |
| **DropPath** | Drop entire residual paths | ViT, modern CNNs |
