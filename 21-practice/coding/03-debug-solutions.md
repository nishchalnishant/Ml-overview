---
module: Practice
topic: Coding Drills
subtopic: Debug Solutions
status: unread
tags: [practice, coding, debugging, solutions]
---
# Debug Drills — Solutions

Drills: [03-debug-drills.md](03-debug-drills.md).

Each entry gives the bug, why it survives casual review, the fix, and the symptom you
would actually see in production.

---

### D1 — Scaling before splitting

**Bug:** `scaler.fit_transform(X)` runs on the full dataset, so the mean and standard
deviation absorb the test rows.

**Why it survives review:** it looks like tidy preprocessing, and the accuracy it reports is
*higher*, so nothing feels broken.

**Fix:**

```python
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2)
scaler = StandardScaler().fit(X_tr)
X_tr, X_te = scaler.transform(X_tr), scaler.transform(X_te)
```

Better: put the scaler in a `Pipeline` so it cannot be gotten wrong.

**Symptom:** offline accuracy consistently overstates production accuracy by a small,
stubborn margin that no amount of tuning closes. Worse on small datasets.

### D2 — Missing `zero_grad`

**Bug:** gradients accumulate across every batch, because PyTorch adds to `.grad` rather than
replacing it. Step *k* is taken on the sum of gradients from batches 1 through *k*.

**Fix:** `optimizer.zero_grad()` before `loss.backward()`.

**Symptom:** loss diverges, or oscillates and refuses to converge, and the effect worsens as
the epoch progresses. Also: the printed loss is the *last batch's*, not the epoch average —
a second, milder bug worth naming.

### D3 — Wrong denominator

**Bug:** `len(loader)` is the number of *batches*, not the number of samples. With batch size
32 the reported accuracy is roughly 32× too high.

**Fix:**

```python
correct = total = 0
for xb, yb in loader:
    ...
    total += yb.size(0)
return correct / total
```

Also missing `model.eval()` and `torch.no_grad()`.

**Symptom:** accuracy above 1.0, which is obvious — unless batch size is 1 in your test
harness and large in production, in which case it silently passes tests.

### D4 — Rolling window includes the current row

**Bug:** `rolling(7)` is inclusive of the current row, so `rolling_mean_7d` for a transaction
contains that transaction's own amount. Since fraud amounts are unusual, the feature leaks the
label.

**Fix:** `s.shift(1).rolling(7).mean()`, or `rolling(7, closed="left")`. Also confirm the
frame is sorted by `(user_id, timestamp)` first — `groupby` does not sort within groups.

**Symptom:** spectacular offline AUC, near-random production performance. The classic
time-series leak, and the single most common cause of "the model worked in the notebook."

### D5 — Softmax before CrossEntropyLoss

**Bug:** `nn.CrossEntropyLoss` applies `log_softmax` internally and expects raw logits.
Applying softmax in `forward` means softmax runs twice.

**Why it survives review:** it still trains. The double-softmax compresses the logit range, so
gradients shrink — the model learns, just badly.

**Fix:** return `self.fc2(x)` and apply softmax only when you need probabilities at inference.

**Symptom:** training is slow to converge and plateaus below what the architecture should
reach. No error, no warning.

### D6 — Shuffled k-fold on temporal data

**Bug:** `shuffle=True` puts future transactions in the training fold and past ones in
validation. The model is trained on the future to predict the past.

**Fix:** `TimeSeriesSplit`, or the walk-forward splitter with a gap from
[01-ml-solutions.md](01-ml-solutions.md#42-walk-forward-split-with-a-gap). The gap matters
whenever the label has a lookahead horizon — here, next-day churn means the last day of
training overlaps the validation label window.

**Symptom:** CV scores far above what production ever reproduces, with the gap unexplainable
by distribution shift.

### D7 — Retaining the autograd graph

**Bug:** `losses.append(loss)` stores a tensor that still holds a reference to its entire
computation graph. None of it can be freed. Ten thousand steps means ten thousand retained
graphs.

**Fix:** `losses.append(loss.item())` — or `loss.detach()` if you need the tensor.

**Symptom:** memory climbs monotonically until OOM, typically hours into a long run, with no
correlation to batch size. One of the most common real PyTorch memory leaks.

### D8 — Ascending sort on similarity

**Bug:** `np.argsort` sorts ascending, so `[:10]` returns the ten *least* similar documents.

**Why it survives review:** the pipeline runs, returns ten documents, and they are real
documents. Nothing looks broken unless you read the output.

**Fix:** `np.argsort(-sims)[:10]`, or better `np.argpartition(-sims, 10)[:10]` followed by a
sort of those ten.

**Second bug worth catching:** cosine similarity requires normalized vectors. If
`model.encode` does not L2-normalize, the raw dot product is ranking by magnitude as much as
by direction.

**Symptom:** RAG answers that are confidently, consistently irrelevant.

### D9 — Resampling before cross-validation

**Bug:** SMOTE synthesizes minority examples by interpolating between neighbors, then CV
splits the *resampled* set. A synthetic point in the validation fold was interpolated from
real points now sitting in the training fold — near-duplicates straddle the split.

**Fix:** resample inside the fold only.

```python
from imblearn.pipeline import Pipeline as ImbPipeline
pipe = ImbPipeline([("smote", SMOTE()), ("clf", model)])
scores = cross_val_score(pipe, X, y, cv=5, scoring="f1")
```

`imblearn`'s pipeline applies the sampler to training folds only; sklearn's would not.

**Symptom:** exactly the reported one — 0.94 offline, 0.31 in production.

### D10 — Batch-dependent padding

**Bug:** `padding=True` pads to the longest sequence *in the current batch*. A short input
batched with a long one gets a different padded length than the same input predicted alone.
Any component that does not strictly honor the attention mask — mean-pooling over the full
padded length is the usual culprit — then produces a different representation for identical
input depending on its neighbors.

**Fix:** mask before any pooling, so pad positions contribute nothing:

```python
mask = tokens["attention_mask"].unsqueeze(-1)
pooled = (hidden * mask).sum(1) / mask.sum(1)
```

For bit-exact reproducibility regardless of batching, use `padding="max_length"` and accept
the wasted compute.

**Symptom:** exactly the reported one — predictions change depending on what else is in the
batch. From the caller's side the API looks nondeterministic, and it is nearly impossible to
reproduce from a single logged request.

### D11 — Tuning on the test set

**Bug:** the test set selects the hyperparameters, so `best_score` is not an unbiased estimate
of anything. With a large enough grid you are fitting the test set through the search loop.

**Fix:** three-way split, or nested CV — select on a validation set (or inner CV), report on a
test set touched exactly once.

**Symptom:** reported accuracy that never reproduces on genuinely new data. The optimism grows
with grid size, so the more thoroughly you search, the more you lie to yourself.

### D12 — Summing an already-averaged loss

**Bug:** `batch_loss` is already the mean over the batch (PyTorch loss functions default to
`reduction="mean"`). Adding it to `total` while dividing by total *samples* divides a sum of
per-sample-averages by the sample count.

**Fix:**

```python
self.total += batch_loss * batch_size
```

**Symptom:** the reported epoch loss is roughly `batch_size` times smaller than any batch
loss — as described. Comparing your training curve to a paper's makes it look like the model
is doing far better than it is.
