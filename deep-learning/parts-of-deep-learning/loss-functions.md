# Loss Functions

Loss functions tell the model what "wrong" means.

That sounds simple.
It is also one of the most important design choices in training.

Because the optimizer can only improve what the loss actually measures.

---

# 1. Classification Losses

## Cross-Entropy

The standard choice for classification.

Why it works well:

- rewards high probability on the correct class
- punishes confident wrong predictions heavily

That last part matters a lot.

## Binary Cross-Entropy

Use for binary classification.

## Multiclass Cross-Entropy

Use for one-of-many class settings.

## Focal Loss

Useful when:

- class imbalance is severe
- easy examples dominate too much

It shifts focus toward harder examples.

---

# 2. Regression Losses

## MSE

Punishes large errors more strongly.

Good when big misses are very costly.

## MAE

More robust to outliers.

Good when you do not want a few extreme points dominating training.

## Huber Loss

Middle ground.

Behaves like:

- MSE for small errors
- MAE for large ones

Very practical.

---

# 3. Match the Loss to the Task

- regression -> MSE / MAE / Huber
- binary classification -> BCE
- multiclass -> cross-entropy
- imbalance-heavy classification -> maybe focal loss

If the loss and output activation do not match the task, training quality suffers fast.

That is a classic interview point.
