# Math Derivations for ML Interviews

---

# 1. Backpropagation Derivation

## Setup

Single neuron forward pass:

$$z = \mathbf{w}^T \mathbf{x} + b, \quad a = \sigma(z), \quad L = \text{Loss}(a, y)$$

## Goal: compute $\partial L / \partial \mathbf{w}$

Apply the chain rule:

$$\frac{\partial L}{\partial \mathbf{w}} = \frac{\partial L}{\partial a} \cdot \frac{\partial a}{\partial z} \cdot \frac{\partial z}{\partial \mathbf{w}}$$

Each factor:

$$\frac{\partial z}{\partial \mathbf{w}} = \mathbf{x}, \quad \frac{\partial a}{\partial z} = \sigma'(z), \quad \frac{\partial L}{\partial a} = \text{depends on loss}$$

Combining:

$$\frac{\partial L}{\partial \mathbf{w}} = \frac{\partial L}{\partial a} \cdot \sigma'(z) \cdot \mathbf{x}$$

Define the **error signal** $\delta = \partial L / \partial z = (\partial L / \partial a) \cdot \sigma'(z)$. Then:

$$\frac{\partial L}{\partial \mathbf{w}} = \delta \cdot \mathbf{x}, \quad \frac{\partial L}{\partial b} = \delta$$

## Multi-layer generalization

For layer $l$ with pre-activation $\mathbf{z}^{(l)} = W^{(l)} \mathbf{a}^{(l-1)} + \mathbf{b}^{(l)}$ and activation $\mathbf{a}^{(l)} = \sigma(\mathbf{z}^{(l)})$:

$$\boldsymbol{\delta}^{(l)} = \left(W^{(l+1)T} \boldsymbol{\delta}^{(l+1)}\right) \odot \sigma'(\mathbf{z}^{(l)})$$

$$\frac{\partial L}{\partial W^{(l)}} = \boldsymbol{\delta}^{(l)} (\mathbf{a}^{(l-1)})^T, \quad \frac{\partial L}{\partial \mathbf{b}^{(l)}} = \boldsymbol{\delta}^{(l)}$$

This recursion propagates the error signal backward from layer $L$ to layer $1$ — hence "backpropagation."

**Computational complexity:** $O(P)$ per sample where $P$ is the number of parameters — same order as the forward pass.

---

# 2. Logistic Regression Cross-Entropy Gradient

## Setup

$$z = \mathbf{w}^T \mathbf{x} + b, \quad \hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}}$$

Binary cross-entropy loss:

$$L = -\left[y \log \hat{y} + (1 - y) \log(1 - \hat{y})\right]$$

## Derivation

**Step 1:** $\partial L / \partial \hat{y}$:

$$\frac{\partial L}{\partial \hat{y}} = -\frac{y}{\hat{y}} + \frac{1-y}{1-\hat{y}}$$

**Step 2:** $\partial \hat{y} / \partial z$ — sigmoid derivative:

$$\frac{\partial \hat{y}}{\partial z} = \sigma(z)(1 - \sigma(z)) = \hat{y}(1 - \hat{y})$$

**Step 3:** chain rule for $\partial L / \partial z$:

$$\frac{\partial L}{\partial z} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z} = \left(-\frac{y}{\hat{y}} + \frac{1-y}{1-\hat{y}}\right) \cdot \hat{y}(1-\hat{y})$$

Expanding:

$$= -y(1 - \hat{y}) + (1-y)\hat{y} = \hat{y} - y$$

**Step 4:** $\partial L / \partial \mathbf{w} = (\partial L / \partial z) \cdot \mathbf{x}$:

$$\boxed{\frac{\partial L}{\partial \mathbf{w}} = (\hat{y} - y)\mathbf{x}, \quad \frac{\partial L}{\partial b} = \hat{y} - y}$$

The gradient is simply prediction minus target, times the input. The cancellation between the sigmoid derivative and the CE derivative is why cross-entropy pairs so naturally with sigmoid output.

---

# 3. Softmax Gradient

For softmax output $s_i = e^{z_i} / \sum_j e^{z_j}$:

$$\frac{\partial s_i}{\partial z_j} = \begin{cases} s_i(1 - s_i) & i = j \\ -s_i s_j & i \neq j \end{cases}$$

In matrix form, the Jacobian $\partial \mathbf{s} / \partial \mathbf{z} = \text{diag}(\mathbf{s}) - \mathbf{s}\mathbf{s}^T$.

**With cross-entropy loss** $L = -\log s_y$ (true class $y$):

$$\frac{\partial L}{\partial z_i} = s_i - \mathbb{1}[i = y]$$

Same pattern: prediction minus one-hot target. This is why PyTorch's `nn.CrossEntropyLoss` takes raw logits and handles softmax+log internally — the combined gradient is numerically stable and clean.

---

# 4. Attention Gradient (Scaled Dot-Product)

Forward pass:

$$A = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right), \quad \text{out} = AV$$

Backward pass (key steps):

$$\frac{\partial L}{\partial V} = A^T \frac{\partial L}{\partial \text{out}}$$

$$\frac{\partial L}{\partial A} = \frac{\partial L}{\partial \text{out}} V^T$$

$$\frac{\partial L}{\partial (QK^T/\sqrt{d_k})} = \text{softmax\_grad}(A) \odot \frac{\partial L}{\partial A}$$

where softmax backward: $\frac{\partial L}{\partial \mathbf{z}} = \mathbf{s} \odot \left(\frac{\partial L}{\partial \mathbf{s}} - \mathbf{s}^T \frac{\partial L}{\partial \mathbf{s}}\right)$

$$\frac{\partial L}{\partial Q} = \frac{1}{\sqrt{d_k}} \cdot \frac{\partial L}{\partial S} \cdot K, \quad \frac{\partial L}{\partial K} = \frac{1}{\sqrt{d_k}} \cdot \left(\frac{\partial L}{\partial S}\right)^T Q$$

---

# 5. Why $\sqrt{d_k}$ Scaling

Without scaling, the variance of $q^T k$ grows with dimension:

$$\text{Var}(q^T k) = \sum_{i=1}^{d_k} \text{Var}(q_i k_i) = d_k \cdot \text{Var}(q_i)\text{Var}(k_i)$$

If $q_i, k_i \sim \mathcal{N}(0, 1)$: $\text{Var}(q^T k) = d_k$.

Large logits push softmax into near-zero gradient regions:

$$\text{softmax}(z_i) \approx \mathbb{1}[i = \arg\max] \implies \text{gradient} \approx 0$$

Dividing by $\sqrt{d_k}$ normalizes variance to 1 regardless of dimension.

---

# 6. Adam Bias Correction Derivation

First moment at step $t$ (unrolling the recursion):

$$m_t = (1 - \beta_1) \sum_{i=1}^{t} \beta_1^{t-i} g_i$$

Expected value (assuming stationary gradients $\mathbb{E}[g_i] = g$):

$$\mathbb{E}[m_t] = (1 - \beta_1) g \sum_{i=1}^{t} \beta_1^{t-i} = g \cdot (1 - \beta_1^t)$$

So $m_t$ underestimates $g$ by factor $(1 - \beta_1^t)$. The bias-corrected estimate:

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$$

satisfies $\mathbb{E}[\hat{m}_t] = g$.

Same derivation applies to the second moment $v_t$ with $\beta_2$.

At $t=1$: $1 - \beta_1^1 = 0.1$, so without correction $m_1 = 0.1 g_1$ — a $10\times$ underestimate when $\beta_1 = 0.9$.

---

# 7. MSE vs BCE for Classification

**Why not MSE for classification?**

For sigmoid output with MSE loss $L = \frac{1}{2}(\hat{y} - y)^2$:

$$\frac{\partial L}{\partial \mathbf{w}} = (\hat{y} - y) \cdot \hat{y}(1 - \hat{y}) \cdot \mathbf{x}$$

When the model is confidently wrong ($\hat{y} \approx 0$, $y = 1$): $\hat{y}(1-\hat{y}) \approx 0$ — **gradient vanishes even when the error is maximum.**

With BCE:

$$\frac{\partial L}{\partial \mathbf{w}} = (\hat{y} - y) \cdot \mathbf{x}$$

The sigmoid derivative cancels out, giving a strong gradient signal regardless of how wrong the prediction is.

---

# 8. Batch Normalization Forward and Backward

## Forward

Given mini-batch $\{x_1, \ldots, x_m\}$:

$$\mu_B = \frac{1}{m} \sum_{i=1}^m x_i, \quad \sigma_B^2 = \frac{1}{m} \sum_{i=1}^m (x_i - \mu_B)^2$$

$$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}, \quad y_i = \gamma \hat{x}_i + \beta$$

## Backward

$$\frac{\partial L}{\partial \hat{x}_i} = \frac{\partial L}{\partial y_i} \cdot \gamma$$

$$\frac{\partial L}{\partial \sigma_B^2} = \sum_i \frac{\partial L}{\partial \hat{x}_i} \cdot (x_i - \mu_B) \cdot \left(-\frac{1}{2}\right)(\sigma_B^2 + \epsilon)^{-3/2}$$

$$\frac{\partial L}{\partial x_i} = \frac{\partial L}{\partial \hat{x}_i} \cdot \frac{1}{\sqrt{\sigma_B^2 + \epsilon}} + \frac{\partial L}{\partial \sigma_B^2} \cdot \frac{2(x_i - \mu_B)}{m} + \frac{\partial L}{\partial \mu_B} \cdot \frac{1}{m}$$

The gradient flows through both the normalization and the mean/variance statistics — this is why BatchNorm smooths the loss landscape.

---

# 9. L2 Regularization Gradient

Total loss: $L_{\text{total}} = L_{\text{task}} + \frac{\lambda}{2}\|\mathbf{w}\|_2^2$

Gradient:

$$\frac{\partial L_{\text{total}}}{\partial \mathbf{w}} = \frac{\partial L_{\text{task}}}{\partial \mathbf{w}} + \lambda \mathbf{w}$$

Update rule:

$$\mathbf{w} \leftarrow \mathbf{w} - \eta\left(\frac{\partial L_{\text{task}}}{\partial \mathbf{w}} + \lambda \mathbf{w}\right) = (1 - \eta\lambda)\mathbf{w} - \eta \frac{\partial L_{\text{task}}}{\partial \mathbf{w}}$$

The $(1 - \eta\lambda)$ factor is **weight decay** — each step shrinks weights by a constant fraction before the gradient update. This is why L2 regularization and weight decay are equivalent for standard GD (but not for Adam, where AdamW is needed).

---

# 10. Derivation Template (Whiteboard Guide)

1. **Define forward equations** — what computes what
2. **State the target** — $\partial L / \partial \theta$ for which $\theta$
3. **Write the chain decomposition** — identify each link explicitly
4. **Compute each partial** — one at a time
5. **Combine and simplify** — look for cancellations
6. **Interpret the final form** — what does the gradient signal mean?

Key simplification to watch for: sigmoid derivative $\hat{y}(1-\hat{y})$ cancels with terms in BCE and softmax+CE gradients, producing $\hat{y} - y$.
