# Math Derivations — Whiteboard Reference

Step-by-step derivations for the problems most commonly asked on a whiteboard. For each: state what you're deriving, write the setup, apply chain rule or decomposition, simplify, interpret the result.

---

## 1. Logistic Regression Gradient

**Goal:** derive $\frac{\partial \mathcal{L}}{\partial w}$ for binary cross-entropy loss.

**Setup:**
$$\hat{p} = \sigma(z), \quad z = w^T x + b$$
$$\mathcal{L} = -[y \log \hat{p} + (1-y)\log(1-\hat{p})]$$

**Chain rule:**
$$\frac{\partial \mathcal{L}}{\partial w} = \frac{\partial \mathcal{L}}{\partial \hat{p}} \cdot \frac{\partial \hat{p}}{\partial z} \cdot \frac{\partial z}{\partial w}$$

**Step 1 — loss w.r.t. prediction:**
$$\frac{\partial \mathcal{L}}{\partial \hat{p}} = -\frac{y}{\hat{p}} + \frac{1-y}{1-\hat{p}}$$

**Step 2 — sigmoid derivative:**
$$\frac{\partial \sigma(z)}{\partial z} = \sigma(z)(1 - \sigma(z)) = \hat{p}(1-\hat{p})$$

**Step 3 — combine:**
$$\frac{\partial \mathcal{L}}{\partial z} = \left(-\frac{y}{\hat{p}} + \frac{1-y}{1-\hat{p}}\right) \cdot \hat{p}(1-\hat{p}) = \hat{p} - y$$

**Final result:**
$$\boxed{\frac{\partial \mathcal{L}}{\partial w} = (\hat{p} - y) \cdot x}$$

**Interpretation:** gradient is prediction error times input. Zero gradient when prediction is perfect.

---

## 2. Softmax + Cross-Entropy Gradient

**Goal:** derive gradient of cross-entropy w.r.t. pre-softmax logits $z_i$.

**Setup:**
$$s_i = \frac{e^{z_i}}{\sum_j e^{z_j}}, \quad \mathcal{L} = -\sum_k y_k \log s_k$$

**Softmax Jacobian** (two cases):
$$\frac{\partial s_i}{\partial z_j} = \begin{cases} s_i(1 - s_i) & i = j \\ -s_i s_j & i \neq j \end{cases}$$

**Chain rule:**
$$\frac{\partial \mathcal{L}}{\partial z_j} = -\sum_k y_k \frac{1}{s_k} \frac{\partial s_k}{\partial z_j}$$

**Expand** (split $k=j$ and $k \neq j$):
$$= -y_j \cdot \frac{s_j(1-s_j)}{s_j} - \sum_{k \neq j} y_k \cdot \frac{-s_j s_k}{s_k}$$
$$= -y_j(1-s_j) + \sum_{k \neq j} y_k s_j = -y_j + s_j \sum_k y_k$$

Since $\sum_k y_k = 1$ for one-hot labels:

$$\boxed{\frac{\partial \mathcal{L}}{\partial z_j} = s_j - y_j}$$

**Interpretation:** same form as logistic regression — prediction minus target. Clean because softmax and cross-entropy are conjugate.

---

## 3. Backpropagation — Chain Rule Through a Layer

**Goal:** derive gradients for a dense layer $z = Wx + b$, $a = \text{ReLU}(z)$.

**Given:** upstream gradient $\delta^{(l)} = \frac{\partial \mathcal{L}}{\partial a^{(l)}}$

**Step 1 — gradient through activation:**
$$\frac{\partial \mathcal{L}}{\partial z^{(l)}} = \delta^{(l)} \odot \text{ReLU}'(z^{(l)}), \quad \text{ReLU}'(z) = \mathbb{1}[z > 0]$$

**Step 2 — gradient w.r.t. weights:**
$$\frac{\partial \mathcal{L}}{\partial W^{(l)}} = \frac{\partial \mathcal{L}}{\partial z^{(l)}} \cdot (a^{(l-1)})^T$$

**Step 3 — gradient to propagate backward:**
$$\delta^{(l-1)} = (W^{(l)})^T \cdot \frac{\partial \mathcal{L}}{\partial z^{(l)}}$$

**Why cache activations?** Computing $\partial z / \partial W$ requires $a^{(l-1)}$, which was computed in the forward pass. Must store it during forward to avoid recomputation.

---

## 4. Adam Update — Bias Correction Derivation

**Goal:** show why $\hat{m}_t = m_t / (1 - \beta_1^t)$ corrects the startup bias.

**Setup:** $m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$ with $m_0 = 0$.

**Unrolling the recurrence:**
$$m_t = (1-\beta_1) \sum_{i=1}^{t} \beta_1^{t-i} g_i$$

**Expected value** (assuming $\mathbb{E}[g_i] = \mathbb{E}[g]$ for all $i$):
$$\mathbb{E}[m_t] = \mathbb{E}[g] \cdot (1-\beta_1) \sum_{i=1}^{t} \beta_1^{t-i} = \mathbb{E}[g] \cdot (1 - \beta_1^t)$$

**Bias correction:**
$$\mathbb{E}\left[\hat{m}_t\right] = \mathbb{E}\left[\frac{m_t}{1-\beta_1^t}\right] = \mathbb{E}[g]$$

**Interpretation:** early in training (small $t$), $\beta_1^t \approx 1$, so $m_t \approx 0$ regardless of gradient. Dividing by $(1 - \beta_1^t)$ removes this startup suppression. After many steps, $\beta_1^t \to 0$ and correction has no effect.

---

## 5. Attention — Scaled Dot-Product

**Goal:** derive the $1/\sqrt{d_k}$ scaling factor.

**Setup:**
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

**Why scale by $\sqrt{d_k}$?**

Let $q, k \in \mathbb{R}^{d_k}$ with components $\sim \mathcal{N}(0, 1)$.

$$q \cdot k = \sum_{i=1}^{d_k} q_i k_i$$

Each term $q_i k_i$ has mean 0, variance 1. Their sum has:
$$\mathbb{E}[q \cdot k] = 0, \quad \text{Var}[q \cdot k] = d_k$$

So $\text{std}[q \cdot k] = \sqrt{d_k}$.

Without scaling, dot products grow as $O(\sqrt{d_k})$. Large dot products push softmax into saturation regions where gradients vanish:

$$\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}} \xrightarrow{z_i \to \infty} 1$$

Dividing by $\sqrt{d_k}$ normalizes variance back to 1, keeping gradients well-conditioned.

---

## 6. Variational Lower Bound (ELBO)

**Goal:** derive the ELBO for VAE training.

**Goal in VAE:** maximize $\log p(x)$ — the log-likelihood of observed data.

**Problem:** $\log p(x) = \log \int p(x|z) p(z) dz$ is intractable.

**Introduce** approximate posterior $q(z|x)$:
$$\log p(x) = \log \int p(x|z) p(z) dz = \log \mathbb{E}_{q(z|x)}\left[\frac{p(x|z) p(z)}{q(z|x)}\right]$$

**Jensen's inequality** ($\log$ is concave):
$$\geq \mathbb{E}_{q(z|x)}\left[\log \frac{p(x|z) p(z)}{q(z|x)}\right]$$

**Expand:**
$$= \underbrace{\mathbb{E}_{q(z|x)}[\log p(x|z)]}_{\text{reconstruction}} - \underbrace{D_{\text{KL}}(q(z|x) \| p(z))}_{\text{KL regularization}}$$

$$\boxed{\text{ELBO} = \mathbb{E}[\log p(x|z)] - D_{\text{KL}}(q(z|x) \| p(z)) \leq \log p(x)}$$

**Interpretation:**
- First term: decoder should reconstruct $x$ well
- Second term: encoder posterior $q(z|x)$ should stay close to the prior $p(z)$
- Maximizing ELBO = maximizing a lower bound on log-likelihood

---

## 7. KL Divergence and Cross-Entropy Connection

**Goal:** show that minimizing cross-entropy = minimizing KL divergence.

**Definitions:**
$$H(p, q) = -\sum_x p(x) \log q(x) \quad \text{(cross-entropy)}$$
$$D_{\text{KL}}(p \| q) = \sum_x p(x) \log \frac{p(x)}{q(x)} = H(p,q) - H(p)$$

**Rewrite:**
$$H(p, q) = D_{\text{KL}}(p \| q) + H(p)$$

Since $H(p)$ is fixed (it's the entropy of the true labels), minimizing $H(p,q)$ is equivalent to minimizing $D_{\text{KL}}(p \| q)$.

**In classification:** $p$ = one-hot label, $q$ = model softmax output. Cross-entropy loss = KL divergence between label distribution and predicted distribution (up to a constant).

---

## 8. Whiteboard Protocol

```
1. Write what you're computing: "I want dL/dW"
2. State the loss and prediction function
3. Apply chain rule: identify the intermediate variables
4. Compute each partial derivative step by step
5. Substitute back, simplify
6. Interpret: "This means gradient is zero when..."
```

**Common mistakes that cost points:**
- Forgetting the chain rule on the activation function
- Getting softmax cross-entropy gradient wrong (it's just $s_j - y_j$, not more complex)
- Not mentioning why you cache activations in backprop
- Skipping the interpretation step — the result alone isn't enough
