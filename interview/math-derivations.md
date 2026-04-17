# Math Derivations

This file is for the whiteboard part of the interview where elegance matters more than showing off every symbol you have ever met.

The rule:

- define the setup
- derive the key expression
- explain why it matters

That is enough to sound sharp.

---

## 1. Backpropagation

If:

- `z = wx + b`
- `a = sigma(z)`
- `L = Loss(a, y)`

Then:

- `dL/dw = dL/da * da/dz * dz/dw`

And since:

- `dz/dw = x`

You get the classic result:

- gradient is upstream error times local input

That is the heart of backprop.

---

## 2. Logistic Regression Gradient

For binary cross-entropy with sigmoid output, the gradient simplifies to:

- `(y_hat - y) * x`

Why this matters:

It is a beautifully clean result and one reason cross-entropy is such a natural loss for classification.

---

## 3. Attention Gradient Intuition

You usually do not need to derive the full attention gradient under pressure.

What matters:

- softmax couples outputs
- scaling by `sqrt(d_k)` keeps scores numerically sane
- attention weights decide how values are mixed

That level of understanding is often enough.

---

## 4. Adam Bias Correction

Adam tracks moving averages that start at zero, so the early estimates are biased low.

Bias correction fixes that startup distortion.

That is the important intuition.

You do not need to perform a full ritual derivation unless specifically asked.
