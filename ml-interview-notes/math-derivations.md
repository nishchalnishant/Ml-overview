# Math Derivations Master Hub

This hub contains step-by-step derivations for the core mathematical foundations of Machine Learning. In interviews, being able to walk through these on a whiteboard demonstrates senior-level depth.

---

# 1. 🔹 Backpropagation (The Chain Rule)

## Q1: How do you derive the weight update for a single hidden layer?

### 🔹 Direct Answer
Backpropagation is the application of the **Chain Rule** to compute the gradient of the Loss function $L$ with respect to every weight $w$:
$$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial a} \cdot \frac{\partial a}{\partial z} \cdot \frac{\partial z}{\partial w}$$
Where $z = wx + b$ and $a = \sigma(z)$.

### 🔹 Deep Dive: Step-by-Step
1. **Forward Pass:** Compute $z = wx+b$, then $a = \sigma(z)$, finally $L = \text{Loss}(a, y)$.
2. **Output Error ($\delta$):** $\frac{\partial L}{\partial z} = (a-y) \cdot \sigma'(z)$.
3. **Weight Gradient:** $\frac{\partial L}{\partial w} = \delta \cdot x$.
4. **Update:** $w = w - \eta \cdot \frac{\partial L}{\partial w}$.

---

# 2. 🔹 Attention Mechanism

## Q2: Derive the Softmax gradient for Self-Attention.

### 🔹 Direct Answer
Self-Attention is computed as: $\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$.
The gradient of $\text{softmax}(x_i) = \frac{e^{x_i}}{\sum e^{x_j}}$ with respect to $x_j$ is:
- $s_i(1-s_i)$ if $i=j$
- $-s_i s_j$ if $i \neq j$

### 🔹 Intuition
The $\sqrt{d_k}$ scaling prevents the dot product from growing too large, which would cause the softmax gradient to vanish (pushing values into the flat region of the sigmoid/softmax).

---

# 3. 🔹 Optimization (Adam)

## Q3: Why do we use Bias Correction in Adam?

### 🔹 Direct Answer
Adam maintains moving averages of the first moment $m_t$ (mean) and second moment $v_t$ (uncentered variance). Since these are initialized to zero, they are biased toward zero, especially during early steps.
**Bias Correction:**
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$
As $t \to \infty$, the denominator $\to 1$ and the correction disappears.

---

# 4. 🔹 Logistic Regression

## Q4: Derive the Cross-Entropy Loss gradient.

### 🔹 Direct Answer
For Logistic Regression, $L = -(y \log(\hat{y}) + (1-y) \log(1-\hat{y}))$ and $\hat{y} = \sigma(z)$.
The gradient simplifies beautifully to:
$$\frac{\partial L}{\partial w} = (\hat{y} - y) \cdot x$$

### 🔹 Deep Dive
The simplicity $(\hat{y}-y)x$ is the reason Cross-Entropy is preferred over MSE for classification—it provides a linear signal for the error, avoiding the vanishing gradient problem of the sigmoid derivative when the prediction is very wrong.

---

## 🔹 Difficulty Tag: 🔴 Hard
