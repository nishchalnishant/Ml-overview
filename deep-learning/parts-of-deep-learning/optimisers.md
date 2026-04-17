# Optimisers

Optimizers are how the model actually moves through the loss landscape.

Same model.
Same data.
Different optimizer.
Very different training experience.

---

# 1. Gradient Descent Family

Core idea:

- compute gradient
- move weights downhill

Variants:

- batch gradient descent
- SGD
- mini-batch SGD

Mini-batch SGD is the practical default for deep learning because it balances:

- speed
- memory
- gradient quality

---

# 2. Momentum

Momentum helps updates keep moving in useful directions instead of zig-zagging too much.

It is especially helpful when the loss surface has:

- ravines
- noisy local geometry

Think of it as giving the optimizer memory of previous movement.

---

# 3. Adam and AdamW

Adam combines:

- momentum-like first-moment tracking
- adaptive per-parameter learning rates

Why it is popular:

- fast iteration
- robust default
- good on many problems

AdamW improves weight decay handling by decoupling it from the adaptive update.

That distinction matters enough to mention.

---

# 4. When SGD Still Matters

Even though Adam is popular, SGD with momentum can still be a strong choice, especially in some vision settings.

Why?

Sometimes it generalizes better after careful tuning.

So the clean answer is not:

> "Adam is always best."

It is:

> "Adam is a great default, but optimizer choice can affect both speed and generalization."

That is the grown-up answer.
