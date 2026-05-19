# Active Learning

---

## The Problem Active Learning Solves

**The problem**: You have 100,000 unlabeled medical images and a radiologist who can label 200 per week. If you randomly pick which 200 to label first, you're wasting labeling budget on easy, redundant examples the model could figure out from nearby samples. You need 5,000 labels before the model becomes useful. With active learning, you might get a useful model at 500 labels — by choosing *which* 500 examples the model actually needs.

**The core insight**: Not all labeled examples are equally informative. A model that is already confident about a region of feature space learns nothing new from another example in that region. The most informative examples are those the model is most uncertain about, or those that are most different from examples it has already seen. If you control which examples get labeled, you should choose the ones that teach the model the most.

**The mechanics**: The active learning loop iterates:
1. Train a model on the current labeled set L.
2. Score all unlabeled examples using a query strategy.
3. Select the top-k most informative unlabeled examples.
4. Send them to a human annotator (oracle).
5. Add the newly labeled examples to L and repeat.

```
Initialize with a small labeled set L (or random sample)
while annotation_budget_remaining:
    train model on L
    score unlabeled pool U using query strategy
    select k examples from U → send to oracle
    add oracle-labeled examples to L
```

**What breaks immediately**: The learning loop assumes the oracle is correct. Label noise — annotators making errors on genuinely ambiguous examples — corrupts the labeled set faster in active learning than in random sampling, because active learning specifically seeks out hard, ambiguous examples that are the most likely to produce disagreement between annotators.

---

## Query Strategies: Uncertainty Sampling

**The problem**: You have a trained model and a pool of unlabeled examples. Which ones should you label next to most improve the model?

**The core insight**: The model is already confident about examples it has effectively learned from. The examples it is *uncertain* about — near the decision boundary, or with split probability mass — are the ones that carry new information.

**The mechanics**: Three formulations of uncertainty:

**Least Confidence**: Choose the example where the top predicted class has the lowest probability. The model is least sure who wins.

$$x^* = \operatorname{argmax}_x (1 - P(\hat{y} \mid x))$$

**Margin Sampling**: Choose the example with the smallest gap between the top two class probabilities. The two most likely classes are nearly tied.

$$x^* = \operatorname{argmin}_x (P(\hat{y}_1 \mid x) - P(\hat{y}_2 \mid x))$$

**Entropy Sampling**: Choose the example that maximizes prediction entropy — the full probability distribution is most spread out.

$$x^* = \operatorname{argmax}_x -\sum_y P(y \mid x) \log P(y \mid x)$$

Entropy is the most general form — it considers the full distribution, not just the top one or two classes.

```python
import numpy as np

def uncertainty_sampling(probs, strategy='entropy', n_query=10):
    if strategy == 'least_confidence':
        scores = 1 - probs.max(axis=1)
    elif strategy == 'margin':
        sorted_probs = np.sort(probs, axis=1)[:, ::-1]
        scores = -(sorted_probs[:, 0] - sorted_probs[:, 1])   # negative: higher = more uncertain
    elif strategy == 'entropy':
        scores = -np.sum(probs * np.log(probs + 1e-9), axis=1)
    return np.argsort(scores)[-n_query:]
```

**What breaks**: Uncertainty sampling is greedy and myopic. It can repeatedly select from the same small uncertain region of feature space — a cluster of ambiguous examples near one piece of the boundary. The resulting labeled set is redundant and fails to cover the full input distribution. This motivates diversity-aware strategies.

---

## Query by Committee (QbC)

**The problem**: A single model's uncertainty estimate is noisy — its predictions fluctuate with initialization, training order, and hyperparameters. If the committee of "possible models consistent with the current training data" disagrees about a prediction, that's a more robust signal of genuine uncertainty.

**The core insight**: Train a committee of different models. The examples they disagree most on are the ones where the current labeled data is insufficient to determine the answer — these are the most valuable to label.

**The mechanics**: Train K models (e.g., bootstrap samples of the training data, or different random initializations). For each unlabeled example, compute the disagreement among the K models' predictions. Query the example with the highest disagreement.

Vote entropy (average distribution across committee members):

$$H = -\sum_y \frac{V(y)}{|C|} \log \frac{V(y)}{|C|}$$

where $V(y)$ = number of committee members voting for class $y$.

```python
# committee = list of trained models
committee_preds = np.stack([m.predict_proba(X_pool) for m in committee])  # (K, n_pool, n_classes)
avg_preds = committee_preds.mean(axis=0)                                   # (n_pool, n_classes)
vote_entropy = -np.sum(avg_preds * np.log(avg_preds + 1e-9), axis=1)
query_idx = np.argsort(vote_entropy)[-n_query:]
```

**What breaks**: Training and maintaining K models multiplies cost by K. For ensembles of deep models, this is often prohibitive. Committee diversity matters — a committee of nearly identical models gives nearly identical votes, making disagreement a weak signal.

---

## Expected Model Change

**The problem**: Uncertainty tells you where the model is confused. But maximum confusion doesn't always mean maximum learning — a confused prediction on an easy, well-represented region teaches the model less than a new example from an underrepresented region.

**The core insight**: Select the example whose labeling would cause the greatest change to the model. If adding an example with any plausible label would force a large parameter update, that example is genuinely valuable for training.

**The mechanics**: Approximate "model change" by the expected magnitude of the gradient update if this example were labeled:

$$x^* = \operatorname{argmax}_x \mathbb{E}_{y \sim P(\cdot|x)} \|\nabla_\theta \mathcal{L}(x, y, \theta)\|$$

For each unlabeled example, estimate the gradient using the current model's predicted probability distribution as a proxy for the label distribution. Select examples with the largest expected gradient norm.

**What breaks**: Computing the full gradient norm requires a backward pass through the model for every unlabeled example — computationally expensive for large models and large pools. Practical only for small networks or linear models where gradient computation is cheap.

---

## Core-Set Methods

**The problem**: Uncertainty sampling ignores whether the newly selected examples are *diverse*. You might label 50 variations of the same ambiguous example from one dense cluster. The model still has no information about large swathes of input space.

**The core insight**: Coverage. Select the subset of unlabeled examples that best covers the entire input space — minimizing the maximum distance from any unlabeled point to its nearest labeled point. This ensures the model has seen examples from every region.

**The mechanics**: Greedy k-center algorithm:
1. Start with the current labeled set.
2. Find the unlabeled point furthest from any labeled point (maximum minimum distance).
3. Label it. Repeat.

$$x^* = \operatorname{argmax}_{x \in U} \min_{x_i \in L} \text{dist}(x, x_i)$$

**What breaks**: Core-Set requires a meaningful distance metric in the feature space. For raw tabular features with mixed scales, Euclidean distance is poorly calibrated — always normalize first. For complex data types (images, text), distances should be computed in an embedding space (a pretrained feature extractor). The greedy algorithm is O(n × k) distance computations per round, which is expensive for large pools.

---

## BADGE — Batch Active Learning by Diverse Gradient Embeddings

**The problem**: You want to query a batch of k examples simultaneously, not one at a time. Selecting k examples purely by uncertainty produces a redundant batch — all k examples cluster around the same uncertain region. Core-Set picks diverse examples but ignores informativeness. You need both.

**The core insight**: Represent each unlabeled example as its gradient embedding — the gradient of the loss with respect to the final layer weights, evaluated at the current model's prediction. This embedding captures both uncertainty (gradient magnitude is high when the model is unsure) and the direction of change (which parameter region the example would update). Then select a diverse batch in this gradient embedding space.

**The mechanics**:
1. Compute gradient embedding for each unlabeled example: $g_x = \nabla_\theta \mathcal{L}(x, \hat{y}, \theta)$ using the model's current predicted label.
2. Run k-means++ initialization on the gradient embeddings to select k diverse examples.

The result is a batch that is both informativeness-weighted (high gradient magnitude) and diverse (spread across different gradient directions).

**What breaks**: Gradient computation for every unlabeled example is expensive. BADGE is primarily practical for moderate pool sizes or when the model has a small final layer (making gradient embeddings tractable). For very large pools, approximate methods (random subsampling of the pool before gradient computation) are necessary.

---

## Stopping Criteria

**The problem**: The active learning loop could run indefinitely. When should you stop?

Three principled stopping points:
- **Fixed budget**: Stop after labeling k examples. Simple and business-aligned.
- **Performance plateau**: Stop when the model improvement per additional label falls below a threshold. Compute a learning curve; stop when the marginal gain diminishes.
- **Uncertainty convergence**: Stop when the maximum uncertainty across the unlabeled pool drops below a threshold. The model has become confident about everything it can see.

---

## Active Learning Settings

| Setting | Description | When to use |
|---|---|---|
| Pool-based | Large unlabeled pool; score all, select top-k | Most common; offline ML pipelines |
| Stream-based | Label or skip each example as it arrives in a stream | Real-time annotation, data collection at inference |
| Membership query | Model can request labels for any input, including synthesized ones | Mostly theoretical; useful for simple hypothesis classes |

---

## Practical Considerations

**Cold start**: With no labeled data, uncertainty sampling can't start. Use random sampling or clustering (k-means on the unlabeled pool, then label one example per cluster) to get initial coverage before the loop begins.

**Batch querying**: In practice, query batches of k=50–100 at once to avoid retraining after every single example. Use diversity-aware batch selection (BADGE, Core-Set) to avoid wasting the batch on redundant examples.

**Calibration matters**: Uncertainty-based strategies rely on the model outputting meaningful probabilities. A poorly calibrated model might output high-confidence predictions on genuinely uncertain inputs. Apply temperature scaling or Platt calibration before using probabilities for query selection.

**Distribution mismatch**: Querying from a biased pool (e.g., only recent data) can still produce a training set that misrepresents important subpopulations. Monitor the label distribution and feature distribution of the actively collected set.

---

## Active Learning with Deep Models

**The problem**: Deep models are expensive to retrain after every annotation round. They also need good uncertainty estimates, which standard deterministic networks don't produce.

**The core insight**: Use MC Dropout or a deep ensemble to get per-example uncertainty estimates without training a full committee. BALD — Bayesian Active Learning by Disagreement — formalizes the information-theoretic criterion: select examples that maximize mutual information between the prediction and the model parameters.

$$x^* = \operatorname{argmax}_x I(y; \theta \mid x, \mathcal{D}) = H(y \mid x, \mathcal{D}) - \mathbb{E}_{p(\theta | \mathcal{D})}[H(y \mid x, \theta)]$$

The first term is the entropy of the expected prediction (high = uncertain). The second term is the expected entropy under each possible model (average confidence across models). BALD selects examples that are uncertain in aggregate but consistent within any single model — these are genuinely epistemically uncertain, not just noisy.

```python
# MC-Dropout uncertainty for BALD approximation
model.train()   # activate dropout
n_samples = 20
preds = np.stack([
    model(X_pool).softmax(-1).detach().numpy() for _ in range(n_samples)
])
# Epistemic uncertainty = total entropy - mean entropy
mean_pred    = preds.mean(0)
total_ent    = -np.sum(mean_pred * np.log(mean_pred + 1e-9), axis=1)
mean_ent     = -np.mean(np.sum(preds * np.log(preds + 1e-9), axis=2), axis=0)
bald_score   = total_ent - mean_ent
query_idx    = np.argsort(bald_score)[-n_query:]
```

---

## Evaluation

The right metric for active learning is the learning curve: model performance on a fixed test set as a function of the number of labeled examples used. Active learning should reach a given performance level with fewer labels than random sampling — the curve should be higher (fewer labels for the same accuracy) than the random baseline.

Always compare against a random sampling baseline. If active learning doesn't outperform random sampling on your problem, the overhead of the query loop isn't justified.
