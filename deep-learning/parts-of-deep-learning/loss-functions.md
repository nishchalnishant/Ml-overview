# Loss Functions

Loss functions define the objective the model minimizes. Choice depends on the task (regression, classification, generation) and desired behavior.

---

## Regression

**Mean squared error (MSE):**
\[
L = \frac{1}{n}\sum_i (y_i - \hat{y}_i)^2
\]
- Differentiable; penalizes large errors heavily. Sensitive to outliers.

**Mean absolute error (MAE):**
\[
L = \frac{1}{n}\sum_i |y_i - \hat{y}_i|
\]
- Robust to outliers; gradient has constant magnitude (except at zero).

**Huber loss:** Quadratic for small \(|y - \hat{y}|\), linear for large; balances MSE and MAE.

---

## Classification

**Binary cross-entropy (BCE):** For \(y \in \{0,1\}\), \(\hat{p} = \sigma(z)\):
\[
L = -\frac{1}{n}\sum_i \bigl( y_i \log \hat{p}_i + (1-y_i)\log(1-\hat{p}_i) \bigr)
\]

**Categorical cross-entropy:** For one-hot labels and softmax logits \(\hat{p}_i = \text{softmax}(z)_i\):
\[
L = -\sum_i y_i \log \hat{p}_i
\]
- Standard for multi-class classification. **Label smoothing:** replace one-hot with \((1-\epsilon)y + \epsilon/K\) to reduce overconfidence.

---

## Language modeling / next-token prediction

**Cross-entropy over vocabulary:** For each position, predict next token; loss is cross-entropy between predicted distribution and target (one-hot token id). Sum or average over positions and batch.

- **Perplexity:** \(\exp(L)\); lower is better. Standard metric for language models.

---

## When to use which

| Task | Loss |
|------|------|
| Regression (default) | MSE |
| Regression (outliers) | MAE or Huber |
| Binary classification | BCE |
| Multi-class classification | Categorical cross-entropy |
| Next-token / LM | Cross-entropy over tokens |
| Contrastive (e.g. SimCLR) | InfoNCE / NT-Xent |

---

## Quick revision

- **MSE:** smooth, sensitive to outliers. **MAE:** robust. **BCE / cross-entropy:** classification and language modeling. **Label smoothing** and **Huber** are common refinements.
