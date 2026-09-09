---
module: Practice
topic: Coding Drills
subtopic: ML Problems
status: unread
tags: [practice, coding, ml, drills]
---
# ML Implementation Problems

**Solutions:** [01-ml-solutions.md](01-ml-solutions.md) — do not open until you have attempted.

Each problem lists a time box. The time box is part of the problem. If you routinely need
double it, that is the finding — note it and move on rather than stretching the clock.

Ground rules that mirror a real screen: NumPy is allowed, scikit-learn is not (unless the
problem says otherwise), and you must handle edge cases without being reminded.

---

## Tier 1 — Warm-ups (15 min each)

**1.1 Numerically stable softmax.**
Implement `softmax(x)` for a 1-D array, then extend to 2-D with a row-wise `axis` argument.
Your implementation must not overflow on `x = [1000, 1001, 1002]`.

**1.2 Train/test split from scratch.**
Implement `train_test_split(X, y, test_size=0.2, seed=None)` without sklearn. Shuffle
correctly, preserve the X–y correspondence, and make the seed produce reproducible output.

**1.3 Standardization with the fit/transform split.**
Write a `StandardScaler` class with `fit`, `transform`, and `fit_transform`. State in a
comment why `fit` must never see the test set, and handle a zero-variance column without
producing NaN.

**1.4 Confusion matrix and derived metrics.**
Given `y_true` and `y_pred` (binary), return TP/FP/TN/FN, then precision, recall, F1.
Handle the zero-denominator cases explicitly — decide what precision means when nothing was
predicted positive, and defend the choice.

**1.5 One-hot encoding with unseen categories.**
Implement `fit`/`transform` one-hot encoding. `transform` must handle a category never seen
during `fit` without crashing. State your policy for that case.

**1.6 Batch generator.**
Write a generator yielding `(X_batch, y_batch)` of a given size, with optional shuffling per
epoch. Handle the final partial batch, and make `drop_last` an option.

---

## Tier 2 — Core algorithms (25 min each)

**2.1 Linear regression, two ways.**
Implement it via the closed-form normal equation and via gradient descent. Then answer in a
comment: at what point does the closed form stop being the right choice, and why?

**2.2 Logistic regression with L2.**
Full implementation: sigmoid, binary cross-entropy loss, gradient, and a fit loop with a
regularization term. Your sigmoid must not overflow on large negative input.

**2.3 K-means with k-means++ init.**
Implement Lloyd's algorithm, then replace random initialization with k-means++. Handle the
empty-cluster case — it will happen and naive implementations crash or silently corrupt.

**2.4 K-nearest neighbours.**
Implement `predict` for classification with configurable `k` and a distance metric argument.
Break ties deterministically and say how in a comment.

**2.5 Decision tree — a single split.**
Given a feature matrix and labels, find the split (feature index, threshold) that maximizes
information gain using Gini. You do not need to build the full tree; the split-finding is
what gets asked.

**2.6 Naive Bayes for text.**
Multinomial NB with Laplace smoothing: fit on tokenized documents, predict a class. Work in
log space and explain why in a comment.

**2.7 PCA from scratch.**
Center the data, compute the covariance matrix, eigendecompose, project onto the top-k
components. Return the explained-variance ratio too. Get the eigenvector ordering right.

**2.8 ROC-AUC without sklearn.**
Compute AUC from scores and binary labels. Handle tied scores correctly — this is the part
that separates a correct implementation from a nearly-correct one.

**2.9 Stratified k-fold.**
Yield train/test index arrays for k folds, preserving class proportions in each fold.
Handle a class with fewer members than `k`.

**2.10 Gradient descent variants.**
Implement vanilla GD, momentum, and Adam as three functions with the same signature,
optimizing a supplied loss. Adam must include bias correction — omitting it is the classic
error.

---

## Tier 3 — Deep learning (30 min each)

**3.1 Backprop through a two-layer network.**
Forward and backward pass in NumPy for `input → linear → ReLU → linear → softmax → CE`.
Return gradients for all four parameter tensors. Verify against a numerical gradient check.

**3.2 Scaled dot-product attention.**
Implement `attention(Q, K, V, mask=None)` in NumPy. Include the `1/sqrt(d_k)` scaling and
explain in a comment what breaks without it. Support an additive causal mask.

**3.3 Multi-head attention.**
Extend 3.2 to `h` heads with the reshape/transpose plumbing. Getting the shapes right *is*
the exercise.

**3.4 Layer normalization, forward and backward.**
Forward is easy; the backward pass is the real problem. Derive it rather than looking it up.

**3.5 A complete PyTorch training loop.**
Write one from memory: dataloader, forward, loss, `zero_grad`, backward, step, eval mode
under `no_grad`, and a per-epoch validation metric. Every omission an interviewer notices is
a point lost — `model.eval()` and `zero_grad()` are the two most commonly forgotten.

**3.6 Dropout at train and inference time.**
Implement inverted dropout. State what changes at inference and why the scaling lives in the
training path.

**3.7 Positional encoding.**
Implement sinusoidal encoding, then say in a comment what a learned encoding buys you and
what it costs.

**3.8 Beam search decoding.**
Given a next-token probability function, implement beam search with configurable width and
length normalization. Handle EOS correctly — a beam that has finished must stop extending.

---

## Tier 4 — Applied and messy (35 min each)

**4.1 Leak-free preprocessing pipeline.**
Given a raw dataframe with missing values, categoricals, and numerics, write a pipeline that
imputes, encodes, and scales — with zero leakage across a cross-validation loop. The leak is
easy to introduce and the interviewer is looking for exactly it.

**4.2 Time-series split.**
Implement a walk-forward CV splitter with a configurable gap between train and test.
Explain in a comment why standard k-fold is invalid here.

**4.3 Imbalanced binary classification, end to end.**
Given a 1:200 dataset, produce a model plus the evaluation you would actually trust.
Justify your metric choice in a comment; accuracy is a trap and PR-AUC is usually the answer.

**4.4 Streaming mean and variance.**
Implement Welford's online algorithm. Explain in a comment why the naive
`sum of squares − square of sum` formulation fails numerically.

**4.5 Top-k retrieval by cosine similarity.**
Given a query vector and a matrix of N document vectors, return the top-k indices. Then make
it fast: normalize once, use a heap or `argpartition` rather than a full sort, and say what
you would change at N = 10 million.

**4.6 Sliding-window feature aggregation.**
For an event log of `(user_id, timestamp, value)`, compute per-user rolling aggregates over a
time window without leaking future information into any row.
