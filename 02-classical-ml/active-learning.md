# Active Learning

Active learning reduces labeling cost by letting the model choose which unlabeled examples a human should label next. The model queries the most informative examples — typically those it is most uncertain about.

**Why it matters:** Labeling is expensive. A model trained on 200 strategically chosen examples often outperforms one trained on 1000 randomly selected ones.

---

## The Active Learning Loop

```
Initialize with small labeled set L
Train model on L
While budget not exhausted:
    Score unlabeled pool U using query strategy
    Select top-k examples from U → send to oracle (human annotator)
    Add labeled examples to L
    Retrain model on L
```

---

## Query Strategies

### Uncertainty Sampling

Select examples the current model is least confident about.

**Least Confidence:**
`x* = argmax_x (1 - P(ŷ|x))`  — pick the sample where top predicted class has lowest probability.

**Margin Sampling:**
`x* = argmin_x (P(ŷ₁|x) - P(ŷ₂|x))` — smallest margin between top-2 predicted classes.

**Entropy Sampling:**
`x* = argmax_x -∑_y P(y|x) log P(y|x)` — maximum prediction entropy.

```python
import numpy as np

def uncertainty_sampling(probs, strategy='entropy', n_query=10):
    if strategy == 'least_confidence':
        scores = 1 - probs.max(axis=1)
    elif strategy == 'margin':
        sorted_probs = np.sort(probs, axis=1)[:, ::-1]
        scores = sorted_probs[:, 0] - sorted_probs[:, 1]
        scores = -scores  # lower margin = higher uncertainty
    elif strategy == 'entropy':
        scores = -np.sum(probs * np.log(probs + 1e-9), axis=1)
    return np.argsort(scores)[-n_query:]
```

**Limitation:** Uncertainty alone can select redundant samples from the same uncertain region.

---

### Query by Committee (QbC)

Train a committee of models (e.g., ensemble or models from different checkpoints). Query examples where the committee disagrees most.

**Vote entropy:** `H = -∑_y (V(y)/|C|) log(V(y)/|C|)` where `V(y)` = votes for class y, `|C|` = committee size.

```python
# Committee of models trained on bootstrap samples
committee_preds = np.stack([m.predict_proba(X_pool) for m in committee])  # shape: (n_models, n_pool, n_classes)
avg_preds = committee_preds.mean(axis=0)
vote_entropy = -np.sum(avg_preds * np.log(avg_preds + 1e-9), axis=1)
query_idx = np.argsort(vote_entropy)[-n_query:]
```

**Strengths:** More diverse than single-model uncertainty; natural for ensembles.

---

### Expected Model Change (EML)

Select examples that would cause the greatest update to the model parameters if labeled.

**Gradient magnitude:** proxy = length of expected gradient update `‖∇_θ L(x, ŷ)‖`

Expensive to compute for large models; practical mainly for small networks or linear models.

---

### Core-Set Methods (Greedy)

Select a diverse subset that minimizes the maximum distance from any unlabeled point to the nearest labeled point. The **Core-Set** method (Sener & Savarese, 2018) greedily solves this k-center problem.

```
Repeat n_query times:
    Find x* in U that maximizes min_{x_i in L} dist(x*, x_i)
    Move x* from U to L
```

**Strengths:** Coverage-focused — avoids redundant cluster sampling.  
**Weaknesses:** Computationally expensive on large pools; needs good feature representations.

---

### BADGE (Batch Active Learning by Diverse Gradient Embeddings)

Combines uncertainty (gradient magnitude) and diversity (k-means++ on gradient space). Balances informativeness and diversity without hyperparameter tuning.

---

## Stopping Criteria

- Fixed budget: stop after k labels acquired
- Performance plateau: stop when model improvement per label drops below threshold
- Uncertainty convergence: stop when max uncertainty in pool drops below threshold

---

## Pool-Based vs Stream-Based vs Membership Query

| Setting | Description | Use case |
|---------|-------------|----------|
| Pool-based | Large unlabeled pool, rank and pick | Most common; offline ML |
| Stream-based | Label or skip each example as it arrives | Real-time systems |
| Membership query | Model can query any input, even synthesized | Rare; mostly theoretical |

---

## Practical Considerations

**Cold start:** With no labeled data, use random sampling or clustering to get initial coverage.

**Batch vs sequential querying:** In practice, query batches of `b` examples (e.g., b=50) to avoid retraining after every single example. Use diversity-aware batch selection (BADGE, Core-Set) to avoid redundant batches.

**Model calibration matters:** Uncertainty-based strategies require well-calibrated probabilities. Use temperature scaling or Platt calibration.

**Label noise:** If annotators make errors, anomalous uncertainty may reflect labeler disagreement, not genuine difficulty.

**Distribution mismatch:** Querying from a biased pool can still result in an unrepresentative training set. Importance weighting may be needed.

---

## Active Learning with Deep Models

- Use **MC-Dropout** to estimate uncertainty without an explicit committee.
- Use **deep ensembles** for better-calibrated uncertainty.
- **BALD (Bayesian Active Learning by Disagreement):** query `argmax_x I(y; θ | x, D)` — mutual information between prediction and model parameters.

```python
# MC-Dropout uncertainty
model.train()  # keep dropout on
n_samples = 10
preds = np.stack([model(X_pool).softmax(-1).detach().numpy() for _ in range(n_samples)])
mean_pred = preds.mean(0)
entropy = -np.sum(mean_pred * np.log(mean_pred + 1e-9), axis=1)
```

---

## Evaluation

Measure **learning curves**: plot model performance (e.g., accuracy) vs number of labels used. Active learning should require fewer labels to reach the same performance as random sampling.

Common baseline: **random sampling** — active learning should strictly dominate this curve.

---

## Key Interview Points

- Active learning trades labeling cost for modeling compute.
- Uncertainty sampling is simple and often effective but ignores diversity.
- Core-Set and BADGE are preferred for batch active learning with deep networks.
- Requires a human-in-the-loop (oracle) — automation breaks down the assumption.
- In production, this pattern is common in annotation pipelines (e.g., labeling medical images, customer support tagging).
