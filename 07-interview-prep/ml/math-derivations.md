---
module: Interview Prep
topic: Ml
subtopic: Math Derivations
status: unread
tags: [interviewprep, ml, ml-math-derivations]
---
# Math Derivations for ML Interviews

---

## What This File Is For

Each derivation is preceded by the scenario where knowing it matters — the specific interview context where an interviewer asks you to derive something and what they are testing for. Then comes the derivation. Then the interpretation. Memorizing steps is not the goal; understanding why the result has the form it does is.

---

## 1. Backpropagation

**When this gets asked and why it matters:** An interviewer asks you to derive how gradients flow backward through a two-layer network. They are testing whether you understand that backpropagation is the chain rule applied to a computation graph — not a special algorithm, but a systematic application of calculus. Candidates who say "autograd handles it" without being able to derive it manually cannot reason about why gradients vanish, what happens when activations saturate, or how to debug training instability.

### Setup

Single neuron forward pass:
$$z = \mathbf{w}^T \mathbf{x} + b, \quad a = \sigma(z), \quad L = \text{Loss}(a, y)$$

### Goal: compute $\partial L / \partial \mathbf{w}$

Apply the chain rule:
$$\frac{\partial L}{\partial \mathbf{w}} = \frac{\partial L}{\partial a} \cdot \frac{\partial a}{\partial z} \cdot \frac{\partial z}{\partial \mathbf{w}}$$

Each factor:
$$\frac{\partial z}{\partial \mathbf{w}} = \mathbf{x}, \quad \frac{\partial a}{\partial z} = \sigma'(z), \quad \frac{\partial L}{\partial a} = \text{depends on loss}$$

Combining:
$$\frac{\partial L}{\partial \mathbf{w}} = \frac{\partial L}{\partial a} \cdot \sigma'(z) \cdot \mathbf{x}$$

Define the **error signal** $\delta = \partial L / \partial z = (\partial L / \partial a) \cdot \sigma'(z)$. Then:
$$\frac{\partial L}{\partial \mathbf{w}} = \delta \cdot \mathbf{x}, \quad \frac{\partial L}{\partial b} = \delta$$

### Multi-layer generalization

For layer $l$ with pre-activation $\mathbf{z}^{(l)} = W^{(l)} \mathbf{a}^{(l-1)} + \mathbf{b}^{(l)}$ and activation $\mathbf{a}^{(l)} = \sigma(\mathbf{z}^{(l)})$:

$$\boldsymbol{\delta}^{(l)} = \left(W^{(l+1)T} \boldsymbol{\delta}^{(l+1)}\right) \odot \sigma'(\mathbf{z}^{(l)})$$

$$\frac{\partial L}{\partial W^{(l)}} = \boldsymbol{\delta}^{(l)} (\mathbf{a}^{(l-1)})^T, \quad \frac{\partial L}{\partial \mathbf{b}^{(l)}} = \boldsymbol{\delta}^{(l)}$$

This recursion propagates the error signal backward from layer $L$ to layer $1$.

**Computational complexity:** $O(P)$ per sample where $P$ is the number of parameters — same order as the forward pass.

**What this tells you about vanishing gradients:** The term $\sigma'(\mathbf{z}^{(l)})$ multiplies the error signal at each layer. For sigmoid, $\sigma'(z) = \sigma(z)(1-\sigma(z)) \leq 0.25$. Multiplying by a factor $\leq 0.25$ at each of 10 layers gives $0.25^{10} \approx 10^{-6}$ — the gradient at early layers is essentially zero. ReLU does not have this problem in its active region because $\sigma'(z) = 1$ for $z > 0$, but suffers from dead neurons where $\sigma'(z) = 0$.

---

## 2. Logistic Regression Cross-Entropy Gradient

**When this gets asked and why it matters:** An interviewer asks why cross-entropy is used instead of MSE for classification and then asks you to derive the gradient. The derivation reveals a beautiful cancellation — the sigmoid derivative and the cross-entropy derivative cancel exactly, producing a clean gradient $\hat{y} - y$. This cancellation is not a coincidence: it is why cross-entropy is the conjugate loss for sigmoid output. Understanding this lets you reason about which loss-activation pairs have this property.

### Setup

$$z = \mathbf{w}^T \mathbf{x} + b, \quad \hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}}$$

Binary cross-entropy loss:
$$L = -\left[y \log \hat{y} + (1 - y) \log(1 - \hat{y})\right]$$

### Derivation

**Step 1:** $\partial L / \partial \hat{y}$:
$$\frac{\partial L}{\partial \hat{y}} = -\frac{y}{\hat{y}} + \frac{1-y}{1-\hat{y}}$$

**Step 2:** Sigmoid derivative:
$$\frac{\partial \hat{y}}{\partial z} = \sigma(z)(1 - \sigma(z)) = \hat{y}(1 - \hat{y})$$

**Step 3:** Chain rule for $\partial L / \partial z$:
$$\frac{\partial L}{\partial z} = \left(-\frac{y}{\hat{y}} + \frac{1-y}{1-\hat{y}}\right) \cdot \hat{y}(1-\hat{y})$$

Expanding:
$$= -y(1 - \hat{y}) + (1-y)\hat{y} = -y + y\hat{y} + \hat{y} - y\hat{y} = \hat{y} - y$$

**Step 4:** $\partial L / \partial \mathbf{w} = (\partial L / \partial z) \cdot \mathbf{x}$:

$$\boxed{\frac{\partial L}{\partial \mathbf{w}} = (\hat{y} - y)\mathbf{x}, \quad \frac{\partial L}{\partial b} = \hat{y} - y}$$

**What this tells you:** The gradient is simply prediction minus target, scaled by the input. The sigmoid derivative $\hat{y}(1-\hat{y})$ cancels entirely. This is not a coincidence — cross-entropy is the negative log-likelihood for a Bernoulli distribution, and sigmoid is the canonical link function for Bernoulli GLMs. The cancellation occurs because the activation function and the loss function are conjugates.

---

## 3. Softmax Gradient

**When this gets asked and why it matters:** An interviewer asks you to derive the Jacobian of softmax. This is tested when discussing multi-class classification, because understanding the gradient tells you why PyTorch's `nn.CrossEntropyLoss` takes raw logits rather than softmax outputs — the combined gradient is numerically stable, and the individual gradients are not. The same cancellation that occurs with sigmoid+BCE occurs with softmax+CE.

### Setup

For softmax output $s_i = e^{z_i} / \sum_j e^{z_j}$, we want $\partial s_i / \partial z_j$.

### Derivation

**Case $i = j$:**
$$\frac{\partial s_i}{\partial z_i} = \frac{e^{z_i} \sum_j e^{z_j} - e^{z_i} \cdot e^{z_i}}{(\sum_j e^{z_j})^2} = s_i - s_i^2 = s_i(1 - s_i)$$

**Case $i \neq j$:**
$$\frac{\partial s_i}{\partial z_j} = \frac{0 - e^{z_i} \cdot e^{z_j}}{(\sum_k e^{z_k})^2} = -s_i s_j$$

In matrix form, the Jacobian $\partial \mathbf{s} / \partial \mathbf{z} = \text{diag}(\mathbf{s}) - \mathbf{s}\mathbf{s}^T$.

**With cross-entropy loss** $L = -\log s_y$ (true class $y$):

$$\frac{\partial L}{\partial z_i} = \sum_k \frac{\partial L}{\partial s_k} \cdot \frac{\partial s_k}{\partial z_i}$$

For $k = y$: $\partial L / \partial s_y = -1/s_y$. For $k \neq y$: $\partial L / \partial s_k = 0$.

$$\frac{\partial L}{\partial z_i} = -\frac{1}{s_y} \cdot \frac{\partial s_y}{\partial z_i} = \begin{cases} -\frac{1}{s_y} \cdot s_y(1-s_y) = s_y - 1 & i = y \\ -\frac{1}{s_y} \cdot (-s_y s_i) = s_i & i \neq y \end{cases}$$

Combined: $\partial L / \partial z_i = s_i - \mathbb{1}[i = y]$, i.e., prediction minus one-hot target.

**What this tells you:** Same pattern as sigmoid+BCE. The gradient is prediction minus target — simple to implement, numerically stable (computing log-softmax directly avoids dividing then taking log), and strong regardless of prediction confidence.

---

## 4. Attention Gradient (Scaled Dot-Product)

**When this gets asked and why it matters:** An interviewer asks how gradients flow through the attention mechanism. This is tested for senior roles where the expectation is that you can reason about attention's computational graph. It also underlies understanding why Flash Attention is an efficiency win — it avoids storing the full attention matrix $A$ for the backward pass by recomputing it.

### Forward pass

$$S = \frac{QK^T}{\sqrt{d_k}}, \quad A = \text{softmax}(S), \quad \text{out} = AV$$

### Backward pass

**Gradient through the output:**
$$\frac{\partial L}{\partial V} = A^T \frac{\partial L}{\partial \text{out}}$$
$$\frac{\partial L}{\partial A} = \frac{\partial L}{\partial \text{out}} V^T$$

**Gradient through softmax:** Using the softmax Jacobian $\partial \mathbf{s}/\partial \mathbf{z} = \text{diag}(\mathbf{s}) - \mathbf{s}\mathbf{s}^T$, for each row $a_i$ of $A$:
$$\frac{\partial L}{\partial s_i} = a_i \odot \left(\frac{\partial L}{\partial a_i} - a_i^T \frac{\partial L}{\partial a_i}\right)$$

**Gradient through $Q$ and $K$:**
$$\frac{\partial L}{\partial Q} = \frac{1}{\sqrt{d_k}} \cdot \frac{\partial L}{\partial S} \cdot K, \quad \frac{\partial L}{\partial K} = \frac{1}{\sqrt{d_k}} \cdot \left(\frac{\partial L}{\partial S}\right)^T Q$$

**What this tells you about Flash Attention:** Standard backprop stores $A \in \mathbb{R}^{n \times n}$ for the backward pass. For $n = 100k$ tokens, this is 10 billion floats. Flash Attention recomputes $A$ during the backward pass from the stored softmax statistics, reducing memory from $O(n^2)$ to $O(n)$ at the cost of extra FLOPs — a favorable tradeoff since memory is the bottleneck.

---

## 5. Why $\sqrt{d_k}$ Scaling

**When this gets asked and why it matters:** An interviewer asks why the attention formula divides by $\sqrt{d_k}$. This is a small derivation that reveals whether you understand variance control in high-dimensional spaces — a concept that recurs in weight initialization (He/Xavier), BatchNorm design, and why dropout rates must be adjusted with model size.

### Derivation

Without scaling, the variance of $q^Tk$ grows with dimension:

If $q_i, k_i \sim \mathcal{N}(0, 1)$ independently:
$$\text{Var}(q_i k_i) = \mathbb{E}[q_i^2 k_i^2] - (\mathbb{E}[q_i k_i])^2 = \mathbb{E}[q_i^2]\mathbb{E}[k_i^2] - 0 = 1 \cdot 1 = 1$$

For the dot product $q^Tk = \sum_{i=1}^{d_k} q_i k_i$ (sum of $d_k$ independent terms each with variance 1):
$$\text{Var}(q^T k) = d_k$$

Standard deviation = $\sqrt{d_k}$. For $d_k = 64$, the standard deviation is 8. Logits of magnitude $\pm 8$ push the softmax toward near-one-hot outputs:
$$\text{softmax}(z)_i \approx \mathbb{1}[i = \arg\max] \implies \nabla_z \text{softmax} \approx 0$$

Dividing by $\sqrt{d_k}$ normalizes the variance to 1 regardless of dimension, keeping softmax in its informative gradient region.

---

## 6. Adam Bias Correction

**When this gets asked and why it matters:** An interviewer asks why Adam uses bias-corrected moment estimates. This tests whether you understand that the exponential moving average is initialized at zero, creating a systematic downward bias in early training. Without bias correction, Adam takes very small steps at the beginning because the moment estimates are close to zero rather than close to the true gradient. This matters practically — it is why Adam warms up differently from SGD with explicit learning rate warmup.

### Derivation

First moment update rule:
$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$

Unrolling from $m_0 = 0$:
$$m_t = (1 - \beta_1) \sum_{i=1}^{t} \beta_1^{t-i} g_i$$

Expected value (assuming stationary gradients $\mathbb{E}[g_i] = g$):
$$\mathbb{E}[m_t] = (1 - \beta_1) g \sum_{i=1}^{t} \beta_1^{t-i} = g \cdot (1 - \beta_1) \cdot \frac{1 - \beta_1^t}{1 - \beta_1} = g \cdot (1 - \beta_1^t)$$

So $m_t$ underestimates $g$ by factor $(1 - \beta_1^t)$. The bias-corrected estimate:
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$$
satisfies $\mathbb{E}[\hat{m}_t] = g$.

At $t=1$ with $\beta_1 = 0.9$: $1 - \beta_1^1 = 0.1$, so $m_1 = 0.1 g_1$ — a 10× underestimate without correction.

The same derivation applies to the second moment $v_t$ with $\beta_2$. As $t$ grows large, $\beta_1^t \to 0$ and the bias correction factor approaches 1 — the correction only matters in early training.

---

## 7. MSE vs BCE for Classification

**When this gets asked and why it matters:** An interviewer asks why MSE should not be used for classification with sigmoid outputs. The derivation shows a specific failure mode: the MSE gradient vanishes precisely when the model is most confidently wrong. This is a gradient problem, not a statistical one — it explains a training failure mode that directly affects convergence. Understanding this lets you explain why BCE is the right loss for sigmoid outputs.

### MSE gradient with sigmoid output

For sigmoid output with MSE loss $L = \frac{1}{2}(\hat{y} - y)^2$:

$$\frac{\partial L}{\partial \mathbf{w}} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z} \cdot \frac{\partial z}{\partial \mathbf{w}} = (\hat{y} - y) \cdot \hat{y}(1 - \hat{y}) \cdot \mathbf{x}$$

**When the model is confidently wrong:** $\hat{y} \approx 0$, $y = 1$ (should predict 1, predicts near 0):
- Error: $\hat{y} - y \approx -1$ — large error
- Sigmoid derivative: $\hat{y}(1 - \hat{y}) \approx 0 \cdot 1 = 0$ — near zero

The gradient $\approx -1 \times 0 \times \mathbf{x} = 0$. The model receives essentially no gradient signal despite being maximally wrong.

### BCE gradient with sigmoid output

With BCE loss, the sigmoid derivative cancels exactly:
$$\frac{\partial L}{\partial \mathbf{w}} = (\hat{y} - y) \cdot \mathbf{x}$$

When $\hat{y} \approx 0$, $y = 1$: gradient $\approx -\mathbf{x}$ — strong signal, correctly pointing toward increasing $\hat{y}$.

**What this tells you:** MSE with sigmoid is a bad design because the loss function's derivative and the activation's derivative have canceling saturation behavior. BCE with sigmoid is a good design because the derivatives cancel each other's saturation, producing a clean linear-in-error gradient regardless of prediction confidence.

---

## 8. Batch Normalization Forward and Backward

**When this gets asked and why it matters:** An interviewer asks you to describe what BatchNorm computes and how gradients flow through it. This is tested because BatchNorm's backward pass is notoriously tricky — gradients flow through both the normalization step and the batch statistics (mean and variance), making the gradient of each example depend on all other examples in the batch. Understanding this explains why BatchNorm performs differently in small batches and why it cannot be used straightforwardly for certain architectures.

### Forward

Given mini-batch $\{x_1, \ldots, x_m\}$:
$$\mu_B = \frac{1}{m} \sum_{i=1}^m x_i, \quad \sigma_B^2 = \frac{1}{m} \sum_{i=1}^m (x_i - \mu_B)^2$$
$$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}, \quad y_i = \gamma \hat{x}_i + \beta$$

Learnable parameters: $\gamma$ (scale) and $\beta$ (shift) allow the network to undo the normalization if that is optimal.

### Backward

$$\frac{\partial L}{\partial \hat{x}_i} = \frac{\partial L}{\partial y_i} \cdot \gamma$$

$$\frac{\partial L}{\partial \sigma_B^2} = \sum_i \frac{\partial L}{\partial \hat{x}_i} \cdot (x_i - \mu_B) \cdot \left(-\frac{1}{2}\right)(\sigma_B^2 + \epsilon)^{-3/2}$$

$$\frac{\partial L}{\partial \mu_B} = \sum_i \frac{\partial L}{\partial \hat{x}_i} \cdot \frac{-1}{\sqrt{\sigma_B^2 + \epsilon}} + \frac{\partial L}{\partial \sigma_B^2} \cdot \frac{-2}{m}\sum_i(x_i - \mu_B)$$

$$\frac{\partial L}{\partial x_i} = \frac{\partial L}{\partial \hat{x}_i} \cdot \frac{1}{\sqrt{\sigma_B^2 + \epsilon}} + \frac{\partial L}{\partial \sigma_B^2} \cdot \frac{2(x_i - \mu_B)}{m} + \frac{\partial L}{\partial \mu_B} \cdot \frac{1}{m}$$

**What this tells you:** The gradient of $x_i$ depends on the gradients of all other $x_j$ through the shared batch statistics $\mu_B$ and $\sigma_B^2$. This is why BatchNorm makes examples within a batch dependent — the effective batch size matters for training stability. With batch size 1, the statistics are computed from a single example and BatchNorm reduces to no normalization. This is why LayerNorm (which normalizes across features, not across the batch) is preferred for sequence models and small-batch training.

---

## 9. L2 Regularization as Weight Decay

**When this gets asked and why it matters:** An interviewer asks about the difference between L2 regularization and weight decay, or asks why AdamW exists separately from Adam. The derivation shows they are equivalent for standard SGD but not for adaptive optimizers — a subtle but important distinction that has practical consequences for training large models.

### Derivation

Total loss: $L_{\text{total}} = L_{\text{task}} + \frac{\lambda}{2}\|\mathbf{w}\|_2^2$

Gradient:
$$\frac{\partial L_{\text{total}}}{\partial \mathbf{w}} = \frac{\partial L_{\text{task}}}{\partial \mathbf{w}} + \lambda \mathbf{w}$$

Update rule with learning rate $\eta$:
$$\mathbf{w} \leftarrow \mathbf{w} - \eta\left(\frac{\partial L_{\text{task}}}{\partial \mathbf{w}} + \lambda \mathbf{w}\right) = (1 - \eta\lambda)\mathbf{w} - \eta \frac{\partial L_{\text{task}}}{\partial \mathbf{w}}$$

The $(1 - \eta\lambda)$ factor is weight decay — each step shrinks weights by a constant fraction before the gradient update.

**For SGD:** L2 regularization and weight decay are equivalent because the gradient of the regularizer enters the update identically to a direct weight decay factor.

**For Adam:** L2 regularization is not equivalent to weight decay. Adam normalizes gradients by the running second moment estimate $\sqrt{v_t}$. The regularizer gradient $\lambda \mathbf{w}$ is normalized along with the task gradient — meaning parameters with large gradient histories receive less effective regularization. AdamW decouples the weight decay from the gradient normalization, applying weight decay directly:
$$\mathbf{w} \leftarrow (1 - \eta\lambda)\mathbf{w} - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

This gives consistent effective regularization regardless of each parameter's gradient history.

---

## Derivation Template (Whiteboard Guide)

1. **Define forward equations** — what computes what, in order
2. **State the target** — $\partial L / \partial \theta$ for which $\theta$
3. **Write the chain decomposition** — identify each link explicitly before computing
4. **Compute each partial** — one at a time, simplest to hardest
5. **Combine and simplify** — look for cancellations
6. **Interpret the final form** — what does the gradient signal mean? When is it large/small/zero?

Key simplification to watch for: sigmoid derivative $\hat{y}(1-\hat{y})$ cancels with terms in BCE and softmax+CE gradients, producing $\hat{y} - y$. This always happens when the activation and the loss are conjugates.

## Rapid Recall

### Error: $\hat{y} - y \approx -1$
- Direct Answer: large error
- Why: This matters because it tells you how to reason about error: $\hat{y} - y \approx -1$.
- Pitfall: Don't answer "Error: $\hat{y} - y \approx -1$" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: large error

### Sigmoid derivative: $\hat{y}(1 - \hat{y}) \approx 0 \cdot 1 = 0$
- Direct Answer: near zero
- Why: This matters because it tells you how to reason about sigmoid derivative: $\hat{y}(1 - \hat{y}) \approx 0 \cdot 1 = 0$.
- Pitfall: Don't answer "Sigmoid derivative: $\hat{y}(1 - \hat{y}) \approx 0 \cdot 1 = 0$" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: near zero
