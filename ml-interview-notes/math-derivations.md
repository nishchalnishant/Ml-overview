# Math Derivations

This file is for whiteboard moments.

Not the terrifying kind where you try to summon every equation you have ever seen.

The useful kind where you explain the derivation cleanly enough that the interviewer can tell:

- you understand the mechanics
- you know what matters
- you are not just reciting symbols from memory

---

# 1. Backpropagation Derivation

## The setup

For a simple neuron:

- `z = wx + b`
- `a = sigma(z)`
- `L = Loss(a, y)`

We want:

- `dL/dw`

## The clean derivation

Use the chain rule:

- `dL/dw = dL/da * da/dz * dz/dw`

Now the pieces:

- `dz/dw = x`
- `da/dz = sigma'(z)`

So:

- `dL/dw = dL/da * sigma'(z) * x`

If you define:

- `delta = dL/dz`

then:

- `dL/dw = delta * x`

That is the key result interviewers usually want.

**Short answer**

Backprop works by chaining local derivatives so the gradient of the loss with respect to a weight becomes the upstream error term times the input feeding that weight.

---

# 2. Logistic Regression Cross-Entropy Gradient

This is a very high-yield derivation.

## Setup

- `z = wx + b`
- `y_hat = sigmoid(z)`

Binary cross-entropy loss:

- `L = -(y log(y_hat) + (1-y) log(1-y_hat))`

## Key result

The derivative simplifies beautifully to:

- `dL/dw = (y_hat - y) * x`

Why this is nice:

The expression becomes much cleaner than you might expect if you differentiate everything naively.

That is one reason cross-entropy pairs so nicely with sigmoid output.

**Strong explanation line**

The gradient simplifies to prediction minus target, times the input, which gives a clean learning signal and avoids some of the awkward behavior you get with MSE on classification.

---

# 3. Softmax Gradient Intuition

You usually do not need to derive the full Jacobian from scratch unless the interviewer really wants it.

What you should know:

For softmax output `s_i`:

- derivative with respect to itself: `s_i(1 - s_i)`
- derivative with respect to another component `s_j`: `-s_i s_j`

Why this matters:

Because softmax outputs are coupled.
Changing one class score affects the others too.

That is the important conceptual point.

---

# 4. Attention Equation

The core attention formula is:

- `Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V`

You should be able to explain each piece:

- `QK^T` = similarity scores
- `sqrt(d_k)` = scaling to keep values numerically sane
- `softmax` = normalized attention weights
- `V` = information being mixed

**Short answer**

Attention computes weighted combinations of value vectors, where the weights come from query-key similarity.

That is the clean explanation.

---

# 5. Why Attention Uses the `sqrt(d_k)` Scaling

Without scaling, dot products can get large as dimension grows.

Large values going into softmax make it too peaky.

That hurts gradient flow and training stability.

So the scaling term keeps the logits in a healthier range.

**Short answer**

We divide by `sqrt(d_k)` so attention scores do not explode with dimension and push softmax into unstable, overly sharp behavior.

---

# 6. Adam Bias Correction

Adam tracks moving averages of:

- first moment `m_t`
- second moment `v_t`

But these moving averages start at zero, so early estimates are biased low.

That is why Adam uses:

- `m_hat_t = m_t / (1 - beta_1^t)`
- `v_hat_t = v_t / (1 - beta_2^t)`

**Short answer**

Bias correction matters because early moving-average estimates are artificially small due to zero initialization, and the correction removes that startup distortion.

---

# 7. Derivation Style That Sounds Good in Interviews

When walking through a derivation:

1. define the variables
2. state what you are solving for
3. write the chain or decomposition clearly
4. simplify only as far as needed
5. explain the meaning of the final expression

That last step matters.

A derivation is not impressive just because it is long.
It is impressive when the endpoint is interpretable.

---

# 8. Quick Whiteboard Templates

## For backprop

- define forward equations
- identify target derivative
- apply chain rule
- name the error term
- give update rule

## For logistic regression

- define sigmoid output
- write BCE loss
- differentiate
- show clean simplification

## For attention

- define Q, K, V
- define score matrix
- explain scaling
- explain weighted sum

These templates help keep you calm.

---

# Quick Thought Experiment

If an interviewer asks:

> "Why is the logistic regression gradient so clean under cross-entropy?"

A strong answer is:

Because the derivative of the sigmoid and the derivative of the cross-entropy combine in a way that simplifies to prediction minus target, which gives a very usable gradient signal.

That is exactly the level of explanation most interviews want.
