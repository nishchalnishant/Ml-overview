---
module: Interview Prep
topic: Llm
subtopic: Math Derivations
status: unread
tags: [interviewprep, ml, llm-math-derivations]
---
# Math Derivations — Whiteboard Reference

For each derivation: state the goal, write the setup, identify the chain rule links, compute each partial derivative, simplify, then *interpret the result*. A derivation that ends with a boxed formula and no interpretation is incomplete.

---

## Whiteboard Protocol

```
1. Announce what you're computing: "I want ∂L/∂W"
2. Write the loss and prediction function explicitly — no ambiguity about notation
3. Name the intermediate variables: identify what the chain rule links
4. Compute each partial derivative separately, left to right
5. Substitute and simplify — look for cancellation
6. Interpret: "This result means the gradient is zero when... and is large when..."
```

---

## 1. Logistic Regression Gradient

### What the interviewer is testing
Whether you can apply the chain rule through a sigmoid without confusion — and whether you recognize that the combined gradient has a remarkably clean form that isn't accidental.

### The derivation

**Goal:** derive $\frac{\partial \mathcal{L}}{\partial w}$ for binary cross-entropy loss with logistic regression.

**Setup:**
$$z = w^T x + b, \quad \hat{p} = \sigma(z) = \frac{1}{1+e^{-z}}, \quad \mathcal{L} = -[y \log \hat{p} + (1-y)\log(1-\hat{p})]$$

**Chain rule — three links in the computational graph:**
$$\frac{\partial \mathcal{L}}{\partial w} = \underbrace{\frac{\partial \mathcal{L}}{\partial \hat{p}}}_{\text{link 1}} \cdot \underbrace{\frac{\partial \hat{p}}{\partial z}}_{\text{link 2}} \cdot \underbrace{\frac{\partial z}{\partial w}}_{\text{link 3}}$$

**Link 1 — loss with respect to prediction:**
$$\frac{\partial \mathcal{L}}{\partial \hat{p}} = -\frac{y}{\hat{p}} + \frac{1-y}{1-\hat{p}}$$

**Link 2 — sigmoid derivative** (worth deriving explicitly):
$$\frac{d\sigma}{dz} = \frac{e^{-z}}{(1+e^{-z})^2} = \frac{1}{1+e^{-z}} \cdot \frac{e^{-z}}{1+e^{-z}} = \sigma(z) \cdot (1 - \sigma(z)) = \hat{p}(1-\hat{p})$$

**Link 3:** $\frac{\partial z}{\partial w} = x$

**Combine links 1 and 2:**
$$\frac{\partial \mathcal{L}}{\partial z} = \left(-\frac{y}{\hat{p}} + \frac{1-y}{1-\hat{p}}\right) \cdot \hat{p}(1-\hat{p})$$

Expand the numerators:
$$= -y(1-\hat{p}) + (1-y)\hat{p} = -y + y\hat{p} + \hat{p} - y\hat{p} = \hat{p} - y$$

**Final result:**
$$\boxed{\frac{\partial \mathcal{L}}{\partial w} = (\hat{p} - y) \cdot x}$$

### Interpretation

The gradient is the *prediction error* $(\hat{p} - y)$ scaled by the input $x$. When the prediction is perfect ($\hat{p} = y$), the gradient is zero — no update, as desired. When you predict $\hat{p} = 1$ but the true label is $y = 0$, the gradient equals $+x$, pushing $w$ in the direction $-x$ to reduce $w^T x$, reducing the probability on the next forward pass.

The clean $(\hat{p} - y)$ form is not a coincidence. Cross-entropy loss and sigmoid are *conjugate* in the exponential family sense — their composition is specifically designed to produce linear-in-error gradients despite the sigmoid nonlinearity. The same structure appears for softmax + categorical cross-entropy.

---

## 2. Softmax + Cross-Entropy Gradient

### What the interviewer is testing
Whether you can handle the softmax Jacobian — specifically the two-case structure ($i = j$ vs $i \neq j$) — and arrive at the clean result under pressure. The Jacobian looks scary; the final answer is simple.

### The derivation

**Goal:** derive $\frac{\partial \mathcal{L}}{\partial z_j}$ for loss $\mathcal{L} = -\sum_k y_k \log s_k$ where $s_i = e^{z_i}/\sum_j e^{z_j}$.

**Step 1 — compute the softmax Jacobian** via quotient rule:

For the numerator $e^{z_i}$ and denominator $\sum_j e^{z_j}$:

Case $i = j$ (differentiating $s_i$ with respect to $z_i$):
$$\frac{\partial s_i}{\partial z_i} = \frac{e^{z_i} \sum_j e^{z_j} - e^{z_i} \cdot e^{z_i}}{(\sum_j e^{z_j})^2} = s_i(1-s_i)$$

Case $i \neq j$ (differentiating $s_i$ with respect to $z_j$, $j \neq i$):
$$\frac{\partial s_i}{\partial z_j} = \frac{0 - e^{z_i} \cdot e^{z_j}}{(\sum_k e^{z_k})^2} = -s_i s_j$$

**Step 2 — apply chain rule:**
$$\frac{\partial \mathcal{L}}{\partial z_j} = -\sum_k \frac{y_k}{s_k} \cdot \frac{\partial s_k}{\partial z_j}$$

**Step 3 — split the sum at $k = j$:**
$$= -\frac{y_j}{s_j} \cdot s_j(1-s_j) \;-\; \sum_{k \neq j} \frac{y_k}{s_k} \cdot (-s_k s_j)$$
$$= -y_j(1-s_j) + s_j \sum_{k \neq j} y_k$$
$$= -y_j + y_j s_j + s_j \sum_{k \neq j} y_k = -y_j + s_j \sum_k y_k$$

For one-hot labels: $\sum_k y_k = 1$, so:

$$\boxed{\frac{\partial \mathcal{L}}{\partial z_j} = s_j - y_j}$$

### Interpretation

Same form as logistic regression: prediction minus target. The Jacobian's complexity cancels entirely. This is not a coincidence — softmax is the canonical link function for the categorical exponential family, and cross-entropy is its conjugate log-likelihood. Their composition always yields $s - y$, by construction.

**Practical implication:** backpropagation through the output layer of a classifier is a single subtraction. No numerical instability from computing softmax probabilities and then dividing again. The "log softmax + NLLLoss" pattern in PyTorch is exactly this: compute log-softmax (numerically stable), then negate the target component. The combined gradient is still $s - y$.

---

## 3. Backpropagation Through a Layer

### What the interviewer is testing
Whether you understand that the backward pass is the chain rule in reverse — and *why* activations must be cached during the forward pass. This is the answer to "why does training memory scale with batch size and sequence length?"

### The derivation

**Setup:** layer $l$ computes $z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)}$, then $a^{(l)} = \text{ReLU}(z^{(l)})$.

**Given:** upstream gradient $\delta^{(l)} = \frac{\partial \mathcal{L}}{\partial a^{(l)}}$ arrives from layer $l+1$.

**Step 1 — gradient through the activation:**
$$\frac{\partial \mathcal{L}}{\partial z^{(l)}} = \delta^{(l)} \odot \text{ReLU}'(z^{(l)}), \qquad \text{ReLU}'(z) = \mathbb{1}[z > 0]$$

The indicator function gates the gradient: neurons that were inactive in the forward pass (negative pre-activation) receive zero gradient. This is why dying ReLU is a permanent failure — a neuron stuck at zero gradient never recovers.

**Step 2 — gradient with respect to the weight matrix:**
$$\frac{\partial \mathcal{L}}{\partial W^{(l)}} = \frac{\partial \mathcal{L}}{\partial z^{(l)}} \cdot (a^{(l-1)})^T$$

This is the outer product of the upstream error signal and the previous layer's activation. *This step requires $a^{(l-1)}$ from the forward pass* — it cannot be recomputed at this stage.

**Step 3 — gradient to propagate backward:**
$$\delta^{(l-1)} = (W^{(l)})^T \cdot \frac{\partial \mathcal{L}}{\partial z^{(l)}}$$

### Why activations are cached — and why this dominates training memory

Step 2 requires $a^{(l-1)}$, which was computed during the forward pass of layer $l-1$. You must store every intermediate activation for every sample in the batch until its layer's backward pass runs. For a Transformer with depth $L$, dimension $d$, sequence length $n$, and batch size $B$: activation memory scales as $O(L \cdot n \cdot d \cdot B)$.

At $L = 32$, $n = 2048$, $d = 4096$, $B = 32$: approximately 32 × 2048 × 4096 × 32 × 2 bytes ≈ 17GB — just for activations, before model parameters or gradients.

**Gradient checkpointing** trades compute for memory: instead of storing all activations, store only the activations at checkpoint boundaries and *recompute* intermediate activations during the backward pass. Roughly doubles forward-pass compute but reduces activation memory from $O(L)$ to $O(\sqrt{L})$ or $O(1)$ depending on checkpoint frequency.

---

## 4. Adam Update — Bias Correction

### What the interviewer is testing
Whether you understand that the bias correction is not a heuristic or a tuning choice — it follows directly from the expectation of a geometric series, and it matters specifically at the *beginning* of training.

### The derivation

**Goal:** show why $\hat{m}_t = m_t / (1 - \beta_1^t)$ is necessary.

**Setup:** first moment estimate initialized to zero — $m_0 = 0$:
$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$$

**Unroll the recurrence:**
$$m_t = (1-\beta_1) \sum_{i=1}^{t} \beta_1^{t-i} g_i$$

**Take the expectation** (assume stationary gradient: $\mathbb{E}[g_i] = g$ for all $i$):
$$\mathbb{E}[m_t] = g \cdot (1-\beta_1) \sum_{i=1}^{t} \beta_1^{t-i} = g \cdot (1-\beta_1^t)$$

**The problem at $t = 1$ with $\beta_1 = 0.9$:**
$$\mathbb{E}[m_1] = 0.1g$$

The estimator is biased toward zero by a factor of $(1-\beta_1^t)$. At step 1, the first moment estimate is 10× smaller than the true gradient direction.

**The bias-corrected estimator:**
$$\hat{m}_t = \frac{m_t}{1-\beta_1^t} \implies \mathbb{E}[\hat{m}_t] = g$$

### Interpretation

Early in training, $1 - \beta_1^t$ is small (at $t=1$: $0.1$; at $t=10$: $1 - 0.9^{10} \approx 0.65$). The uncorrected moment estimate severely underestimates the gradient's magnitude. Dividing by $(1-\beta_1^t)$ inflates early estimates to their correct scale — this is why Adam takes larger steps at the start of training despite the moment being near zero.

As $t \to \infty$: $\beta_1^t \to 0$, so $1 - \beta_1^t \to 1$ and the correction becomes irrelevant. The bias correction only matters during the warmup phase.

Identical analysis applies to the second moment $v_t$ with correction factor $(1 - \beta_2^t)$.

**AdamW vs Adam:** in standard Adam, weight decay is applied as a gradient addition: $g_t \mathrel{+}= \lambda\theta_t$. This gets scaled by the adaptive learning rate $1/\sqrt{\hat{v}_t}$, which means large-gradient parameters experience less weight decay than small-gradient ones — unintended behavior. AdamW applies weight decay directly to parameters: $\theta \mathrel{*}= (1 - \alpha\lambda)$ before the gradient update, fully decoupled from the adaptive scaling. This is the correct implementation for neural networks with weight decay.

---

## 5. Attention — Why the $\sqrt{d_k}$ Scaling

### What the interviewer is testing
Whether you can derive the variance argument from scratch, not just quote the result. This derivation appears in the attention paper and is testable.

### The derivation

**Goal:** show why $\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$ uses $\sqrt{d_k}$ and not some other value.

**Setup:** consider $q, k \in \mathbb{R}^{d_k}$ with components drawn i.i.d. from $\mathcal{N}(0, 1)$.

**Variance of the dot product:**
$$q \cdot k = \sum_{i=1}^{d_k} q_i k_i$$

Each term: $\mathbb{E}[q_i k_i] = \mathbb{E}[q_i]\mathbb{E}[k_i] = 0$ (by independence).
$$\text{Var}(q_i k_i) = \mathbb{E}[q_i^2 k_i^2] - 0 = \mathbb{E}[q_i^2]\mathbb{E}[k_i^2] = 1 \cdot 1 = 1$$

By independence of terms:
$$\text{Var}(q \cdot k) = \sum_{i=1}^{d_k} \text{Var}(q_i k_i) = d_k \implies \text{std}(q \cdot k) = \sqrt{d_k}$$

**What goes wrong without scaling:**

At $d_k = 64$: scores have std 8. A maximum score might be around 30. In softmax:
$$\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

For $z_1 = 30$, $z_2 = 5$: $e^{30}/(e^{30} + e^5) \approx 1$. The softmax approaches a one-hot vector. Its Jacobian:
$$\frac{\partial s_i}{\partial z_j} = s_i(\delta_{ij} - s_j)$$

When $s_i \approx 1$ and $s_{j \neq i} \approx 0$: the cross-terms $s_i s_j \approx 0$. Essentially no gradient flows through the softmax. Attention layer learns nothing.

**The fix:** divide by $\sqrt{d_k}$:
$$\text{Var}\left(\frac{q \cdot k}{\sqrt{d_k}}\right) = \frac{d_k}{d_k} = 1$$

This normalizes variance to 1 regardless of embedding dimension — keeping softmax in its informative, gradient-flowing regime.

### Interpretation

The value $\sqrt{d_k}$ is not a hyperparameter to tune — it is exactly the standard deviation of the dot product under the i.i.d. Gaussian assumption. Dividing by it normalizes to unit variance. The original Transformer paper notes this directly: the scaling exists because the dot products grow with $d_k$ and this pushes softmax into regions of very small gradients.

---

## 6. Variational Lower Bound (ELBO)

### What the interviewer is testing
Whether you can apply Jensen's inequality in the correct direction — and explain why the ELBO is a *lower bound* (not upper) on the log-likelihood.

### The derivation

**Goal:** derive the ELBO — the objective that makes VAE training tractable.

**Starting point:** maximize $\log p_\theta(x)$, the marginal log-likelihood.

**Problem:** $\log p_\theta(x) = \log \int p_\theta(x \mid z) p(z)\, dz$. For neural decoders, this integral over all latent configurations is intractable.

**Step 1 — introduce an approximate posterior** $q_\phi(z \mid x)$:
$$\log p_\theta(x) = \log \int p_\theta(x \mid z) p(z)\, dz = \log \mathbb{E}_{z \sim q_\phi(z \mid x)}\left[\frac{p_\theta(x \mid z) p(z)}{q_\phi(z \mid x)}\right]$$

**Step 2 — apply Jensen's inequality.** For a concave function $f$ (log is concave): $f(\mathbb{E}[X]) \geq \mathbb{E}[f(X)]$. Therefore $\log \mathbb{E}[X] \geq \mathbb{E}[\log X]$:
$$\log p_\theta(x) \geq \mathbb{E}_{q_\phi}\left[\log \frac{p_\theta(x \mid z) p(z)}{q_\phi(z \mid x)}\right]$$

**Step 3 — expand:**
$$= \mathbb{E}_{q_\phi}[\log p_\theta(x \mid z)] + \mathbb{E}_{q_\phi}\left[\log \frac{p(z)}{q_\phi(z \mid x)}\right]$$

$$\boxed{\text{ELBO} = \underbrace{\mathbb{E}_{q_\phi}[\log p_\theta(x \mid z)]}_{\text{reconstruction term}} - \underbrace{D_\text{KL}(q_\phi(z \mid x) \| p(z))}_{\text{regularization term}} \leq \log p_\theta(x)}$$

### Interpretation

- **Reconstruction term:** the decoder must be able to reconstruct $x$ from samples drawn from the encoder's distribution. Maximizing this trains the encoder-decoder pair.
- **KL term:** the encoder's approximate posterior $q_\phi(z \mid x)$ must stay close to the prior $p(z) = \mathcal{N}(0, I)$. This regularizes the latent space — keeps it continuous and sample-able.
- **The gap:** $\log p_\theta(x) - \text{ELBO} = D_\text{KL}(q_\phi(z \mid x) \| p_\theta(z \mid x))$ — how well the encoder approximates the true posterior. Maximizing ELBO simultaneously improves the generative model and tightens this approximation.

The two terms are in necessary tension: the reconstruction term wants the encoder to give an informative $z$ that makes decoding easy; the KL term wants the encoder to ignore $x$ and output the prior. This tension is what forces the latent space to have structure.

---

## 7. KL Divergence — Cross-Entropy Connection

### What the interviewer is testing
Whether you know that minimizing cross-entropy loss is mathematically equivalent to minimizing KL divergence — and can state the exact relationship and what it implies.

### The derivation

**Definitions:**
$$H(p) = -\sum_x p(x) \log p(x) \quad \text{(entropy of true distribution)}$$
$$H(p, q) = -\sum_x p(x) \log q(x) \quad \text{(cross-entropy)}$$
$$D_\text{KL}(p \| q) = \sum_x p(x) \log \frac{p(x)}{q(x)}$$

**Decompose KL:**
$$D_\text{KL}(p \| q) = \sum_x p(x)[\log p(x) - \log q(x)] = -H(p) + H(p,q)$$

**Rearrange:**
$$H(p, q) = D_\text{KL}(p \| q) + H(p)$$

**The key observation:** $H(p)$ is fixed — it depends only on the true label distribution $p$, which does not depend on the model parameters $\theta$. Therefore:
$$\min_\theta H(p, q_\theta) \iff \min_\theta D_\text{KL}(p \| q_\theta)$$

### Interpretation

Training a classifier with cross-entropy loss is equivalent to minimizing the KL divergence between the true label distribution and the model's predicted distribution. The entropy of the labels is a *constant floor* — no model can achieve lower cross-entropy than $H(p)$, the entropy of the true labels. A model with cross-entropy exactly equal to $H(p)$ has perfectly captured the true distribution.

This is why cross-entropy is the "right" loss for classification: it directly minimizes the statistical distance between your model and the true distribution, in a precise information-theoretic sense.

---

## 8. Normal Equation for Linear Regression

### What the interviewer is testing
Whether you can derive the closed-form solution, state its cost, and explain exactly when to use gradient descent instead.

### The derivation

**Goal:** minimize $\mathcal{L}(\theta) = \|X\theta - y\|^2$.

**Expand:**
$$\mathcal{L}(\theta) = (X\theta - y)^T(X\theta - y) = \theta^T X^T X \theta - 2y^T X\theta + y^T y$$

**Set gradient to zero** (using $\nabla_\theta(\theta^T A \theta) = 2A\theta$ for symmetric $A$):
$$\frac{\partial \mathcal{L}}{\partial \theta} = 2X^T X\theta - 2X^T y = 0$$

$$\boxed{\hat{\theta} = (X^T X)^{-1} X^T y}$$

This is the exact, global minimum — linear regression has a convex loss with a unique minimum (assuming $X^T X$ is invertible).

### When not to use it — and exactly why

$(X^T X)$ inversion costs $O(d^3)$ where $d$ is the number of features. At $d = 10{,}000$: $10^{12}$ floating-point operations — infeasible. Gradient descent costs $O(nd)$ per step and scales to arbitrary dimension.

$(X^T X)$ is singular when features are linearly dependent (multicollinearity). The inverse doesn't exist. Fix: ridge regression adds $\lambda I$ to ensure invertibility: $\hat{\theta} = (X^T X + \lambda I)^{-1} X^T y$. This has a probabilistic interpretation: MAP estimation with a Gaussian prior — the $\lambda I$ term is the prior precision.

---

## 9. Common Derivation Mistakes

**Forgetting the chain rule through the activation:**
The gradient through ReLU is $\delta \odot \mathbb{1}[z > 0]$, not just $\delta$. Skipping the element-wise mask produces wrong gradients for every neuron that was inactive in the forward pass.

**Getting the softmax-CE gradient wrong in the exam:**
Under pressure, candidates try to expand the full Jacobian and get confused by the two-case structure. Know in advance that the result is simply $s_j - y_j$, then derive it if asked. The derivation shows the Jacobian mess cancels perfectly — state this confidence before starting the algebra.

**Not explaining why activations are cached:**
Saying "cache activations for backprop" without the why misses the point. The specific reason: $\partial \mathcal{L}/\partial W^{(l)} = \delta^{(l)} \cdot (a^{(l-1)})^T$, and $a^{(l-1)}$ is only available from the forward pass. The backward pass cannot recompute it without re-running the forward pass.

**Deriving without interpreting:**
The formula alone is never a complete answer. After $\partial \mathcal{L}/\partial w = (\hat{p} - y)x$, state: "The gradient is zero when the prediction is perfect; the update direction is the error times the input — the network increases weights in the direction that most influenced the wrong prediction." Interpretation demonstrates understanding; the formula only demonstrates execution.

**Not knowing Adam's bias correction is derived, not assumed:**
The correction factor $(1 - \beta_1^t)$ falls directly out of taking the expectation of the unrolled EMA. It is not a heuristic. At $t = 1$ with $\beta_1 = 0.9$, the raw first moment is 10× too small. The correction inflates it to the correct scale. This is why Adam starts fast.

## Rapid Recall

### Reconstruction term
- Direct Answer: the decoder must be able to reconstruct $x$ from samples drawn from the encoder's distribution. Maximizing this trains the encoder-decoder pair.
- Why: This matters because it tells you how to reason about reconstruction term.
- Pitfall: Don't answer "Reconstruction term" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: the decoder must be able to reconstruct $x$ from samples drawn from the encoder's distribution. Maximizing this trains the encoder-decoder pair.

### KL term: the encoder's approximate posterior $q_\phi(z \mid x)$ must stay close to the prior $p(z) = \mathcal{N}(0, I)$. This regularizes the latent space
- Direct Answer: keeps it continuous and sample-able.
- Why: This matters because it tells you how to reason about kl term: the encoder's approximate posterior $q_\phi(z \mid x)$ must stay close to the prior $p(z) = \mathcal{n}(0, i)$. this regularizes the latent space.
- Pitfall: Don't answer "KL term: the encoder's approximate posterior $q_\phi(z \mid x)$ must stay close to the prior $p(z) = \mathcal{N}(0, I)$. This regularizes the latent space" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: keeps it continuous and sample-able.

### The gap: $\log p_\theta(x) - \text{ELBO} = D_\text{KL}(q_\phi(z \mid x) \| p_\theta(z \mid x))$
- Direct Answer: how well the encoder approximates the true posterior. Maximizing ELBO simultaneously improves the generative model and tightens this approximation.
- Why: This matters because it tells you how to reason about the gap: $\log p_\theta(x) - \text{elbo} = d_\text{kl}(q_\phi(z \mid x) \| p_\theta(z \mid x))$.
- Pitfall: Don't answer "The gap: $\log p_\theta(x) - \text{ELBO} = D_\text{KL}(q_\phi(z \mid x) \| p_\theta(z \mid x))$" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: how well the encoder approximates the true posterior. Maximizing ELBO simultaneously improves the generative model and tightens this approximation.
