# Regularization

Regularization reduces overfitting by constraining or perturbing the model. It includes weight decay, dropout, normalization, and early stopping.

---

## Weight decay (L2 regularization)

**Loss:** \(L_{total} = L_{data} + \frac{\lambda}{2}\|\theta\|^2\). Equivalently, decay weights each step: \(\theta \leftarrow (1-\lambda\eta)\theta - \eta \nabla L\).

- **Effect:** Shrinks weights toward zero; encourages smaller weights and smoother functions.
- **In Adam:** Use **decoupled** weight decay (AdamW), not L2 in the gradient, so decay matches intended regularization.

---

## Dropout

**Training:** For each layer output, zero each element independently with probability \(p\) (e.g. 0.1–0.5); scale remaining by \(1/(1-p)\) to keep expectation unchanged.  
**Inference:** No dropout; use full network.

- **Effect:** Prevents co-adaptation of units; acts as ensemble over sub-networks.
- **Common:** Dropout on attention weights or FFN in transformers (e.g. 0.1); less used in very large pretrained models.

---

## Early stopping

Stop training when validation loss (or metric) stops improving (e.g. no improvement for N epochs). Restore weights from the best validation checkpoint.

- **Effect:** Limits effective capacity by limiting training time; simple and widely used.

---

## Batch normalization (BatchNorm)

**Per dimension (over the batch):** compute mean \(\mu\) and variance \(\sigma^2\) over the batch; normalize; then scale and shift with learnable \(\gamma\), \(\beta\):

\[
\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}, \quad y = \gamma \hat{x} + \beta
\]

- **Placement:** Usually after linear/conv and before activation. **Training:** \(\mu,\sigma^2\) from current batch; running averages kept for inference. **Inference:** use running mean/variance.
- **Benefits:** Faster convergence, allows higher learning rates; reduces internal covariate shift. **Drawbacks:** Depends on batch size; behavior differs at train vs test. Common in CNNs.

---

## Layer normalization (LayerNorm)

**Per sample (over features):** for each sample, compute mean and variance over the feature dimension; normalize; scale and shift:

\[
\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}, \quad y = \gamma \hat{x} + \beta
\]

- **Placement:** In transformers, typically applied before (pre-LN) or after (post-LN) sublayers. **No batch dimension:** same at train and test; stable for variable-length and small batches.
- **Standard in transformers** and LLMs (BERT, GPT, LLaMA, etc.).

---

## Weight initialization

- **Xavier/Glorot:** scale by \(1/\sqrt{n_{in}}\) or \(2/(n_{in}+n_{out})\) so variance of activations is preserved (for tanh/sigmoid).
- **He/Kaiming:** scale by \(\sqrt{2/n_{in}}\) for ReLU (accounts for zero half of distribution).
- **Small random** (e.g. 0.02) for embedding and output layers. Proper init avoids vanishing/exploding gradients in deep networks.

---

## Quick revision

- **Weight decay:** L2 penalty or decoupled decay (AdamW). **Dropout:** randomly zero activations in training; reduces overfitting. **Early stopping:** stop when validation stops improving.
- **BatchNorm:** normalize over batch dimension; common in CNNs. **LayerNorm:** normalize over feature dimension; standard in transformers. **Init:** Xavier/He to keep activation variance stable.
