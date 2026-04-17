# Regularization

Regularization is how you stop a deep model from becoming wildly overconfident about noisy patterns.

It is not about making the model weak.
It is about making it less gullible.

---

# 1. L1 and L2

## L1

Encourages sparsity.

## L2 / Weight Decay

Shrinks weights more smoothly.

In deep learning, weight decay is usually the more common everyday choice.

---

# 2. Dropout

Dropout randomly zeroes activations during training.

Why it helps:

- reduces over-reliance on specific paths
- encourages more robust representations

It is like making sure the whole team can play, not just one star batter.

---

# 3. BatchNorm and Regularization Effect

BatchNorm is mostly about optimization stability, but it can also have a mild regularizing effect.

That is why some architectures use less dropout once normalization is doing part of the stabilizing work.

---

# 4. Early Stopping

Train until validation stops improving.

Very practical.
Very underrated.

Sometimes the best regularization trick is simply not letting the model keep overfitting after the useful learning is already done.

---

# 5. Data Augmentation

Especially important in vision.

Augmentation can act like regularization by teaching useful invariances and exposing the model to broader variation.

This is one of the highest-leverage ways to improve generalization on limited data.
