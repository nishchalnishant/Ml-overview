---
module: Classical ML
topic: Recommender Systems (Classical Methods)
subtopic: ""
status: unread
tags: [classicalml, ml, recommender-systems, collaborative-filtering, matrix-factorization]
---
# Recommender Systems — Classical Methods

> This file covers the **classical, non-neural** foundations: neighborhood-based collaborative filtering and matrix factorization. For neural collaborative filtering, two-tower models, learning-to-rank, cold start, GNN-based recommenders, and production serving patterns, see the full deep-dive at [04-specialized-domains/05-recommender-systems.md](../04-specialized-domains/05-recommender-systems.md).

## Table of Contents

1. [The Setup](#1-the-setup)
2. [Neighborhood-Based Collaborative Filtering](#2-neighborhood-based-collaborative-filtering)
3. [Matrix Factorization](#3-matrix-factorization)
4. [ALS vs SGD](#4-als-vs-sgd)
5. [Implicit Feedback](#5-implicit-feedback)
6. [Evaluation](#6-evaluation)
7. [Canonical Interview Q&As](#canonical-interview-qas)

---

## 1. The Setup

Given users $U$, items $I$, and a sparse interaction matrix $R \in \mathbb{R}^{|U| \times |I|}$ (ratings, clicks, or purchases — usually 95%+ empty), the goal is to predict $\hat{r}_{ui}$ for unobserved pairs and rank items by predicted score. Two classical families solve this without any neural network: **neighborhood methods** (memory-based) and **matrix factorization** (model-based).

## 2. Neighborhood-Based Collaborative Filtering

**User-based CF:** to predict $\hat{r}_{ui}$, find the $k$ users most similar to $u$ who have rated item $i$, and take a similarity-weighted average of their ratings:

$$\hat{r}_{ui} = \bar{r}_u + \frac{\sum_{v \in N_k(u)} \text{sim}(u,v)(r_{vi} - \bar{r}_v)}{\sum_{v \in N_k(u)} |\text{sim}(u,v)|}$$

where similarity is typically cosine similarity or Pearson correlation over the users' co-rated items, and mean-centering ($\bar{r}_u$, $\bar{r}_v$) corrects for users who rate systematically high or low.

**Item-based CF:** the same idea, but similarity is computed between items instead of users, using their co-rating vectors across users. Item-based CF is the more common production choice — item similarity is more stable over time (an item's identity doesn't drift the way a user's taste can day to day), and item-item similarity matrices can be precomputed offline and cached, since the item catalog grows much more slowly than the user base.

This is literally KNN applied in user-space or item-space (cross-reference: [supervised-learning.md](01-supervised-learning.md) §KNN) — the "model" is just the stored interaction matrix plus a similarity function; there's no training step beyond similarity computation.

**Weakness:** the similarity matrix is $O(|U|^2)$ or $O(|I|^2)$ — infeasible to compute exactly at scale (millions of users/items) without approximate nearest-neighbor techniques (LSH, HNSW), and it captures only pairwise co-occurrence, not latent structure shared across many items at once.

## 3. Matrix Factorization

**The idea:** instead of comparing rows/columns of $R$ directly, learn low-rank latent factor vectors $p_u \in \mathbb{R}^k$ (per user) and $q_i \in \mathbb{R}^k$ (per item), $k \ll |U|, |I|$, such that:

$$\hat{r}_{ui} = p_u^\top q_i$$

This is the same low-rank approximation idea as PCA/SVD ([dimensionality-reduction.md](10-dimensionality-reduction.md)) — the difference is $R$ is sparse and only *observed* entries contribute to the loss, so classical full-matrix SVD doesn't directly apply (you can't SVD a matrix with missing entries). Instead, factors are learned by minimizing squared error over observed entries only, plus regularization:

$$\min_{P,Q} \sum_{(u,i) \in \text{observed}} (r_{ui} - p_u^\top q_i)^2 + \lambda(\|p_u\|^2 + \|q_i\|^2)$$

**Biases:** production systems almost always add per-user and per-item bias terms plus a global mean, since raw ratings are dominated by "some users rate everything high" and "some items are universally liked":

$$\hat{r}_{ui} = \mu + b_u + b_i + p_u^\top q_i$$

This bias-augmented form is the core of the **SVD++** / Netflix Prize–style factorization model.

## 4. ALS vs SGD

Two standard ways to optimize the factorization objective:

| Method | How it works | When to prefer |
|---|---|---|
| **SGD** | Sample an observed $(u,i,r_{ui})$, take a gradient step on $p_u$ and $q_i$ | Simple, flexible (easy to add bias terms, regularizers); sequential — harder to parallelize |
| **ALS (Alternating Least Squares)** | Fix $Q$, solve the now-linear least-squares problem for all $P$ in closed form; fix $P$, solve for $Q$; repeat | Each sub-problem is convex and embarrassingly parallel across users/items (row/column-independent) — standard choice for distributed settings (Spark's `ALS` implementation) |

Both converge to a local minimum (the joint objective is non-convex in $(P, Q)$ jointly, convex in each alone) — initialization and $k$ (latent dimension) are the main hyperparameters to tune, alongside $\lambda$.

## 5. Implicit Feedback

Most production data is implicit (clicks, watches, purchases — not explicit 1-5 star ratings), and absence of interaction doesn't mean "dislike," it means "unobserved." The standard fix is **Weighted ALS (WALS / "Hu-Koren-Volinsky")**: convert implicit signal into a confidence-weighted preference,

$$p_{ui} = \mathbb{1}[r_{ui} > 0], \quad c_{ui} = 1 + \alpha r_{ui}$$

and minimize $\sum_{u,i} c_{ui}(p_{ui} - p_u^\top q_i)^2 + \lambda(\ldots)$ over **all** $(u,i)$ pairs (not just observed ones), weighting confident observations (repeated clicks) more heavily while still letting unobserved pairs pull toward 0. This reformulation is what makes ALS tractable at full-matrix scale despite summing over every pair — the weighted least-squares update has a closed form that avoids the $O(|I|)$ or $O(|U|)$ cost per row by exploiting that $C$ is diagonal plus a low-rank correction.

## 6. Evaluation

Rating-prediction metrics (RMSE/MAE) are largely obsolete in production — the Netflix Prize–era objective doesn't correlate well with ranking quality. Modern evaluation uses **ranking metrics** (Precision@K, Recall@K, NDCG@K, MAP) computed on held-out interactions — see [ml-evaluation-metrics.md](12-ml-evaluation-metrics.md) for the ranking-metrics deep dive, and [04-specialized-domains/05-recommender-systems.md](../04-specialized-domains/05-recommender-systems.md) §9-10 for diversity/serendipity/coverage metrics that pure accuracy metrics miss entirely.

---

## Canonical Interview Q&As

**Q: Derive the matrix factorization objective for explicit feedback and explain why plain SVD doesn't work directly.**
A: The goal is to find $P \in \mathbb{R}^{|U|\times k}$, $Q \in \mathbb{R}^{|I|\times k}$ such that $\hat{r}_{ui} = p_u^\top q_i \approx r_{ui}$. Classical SVD would decompose the full matrix $R = U\Sigma V^\top$, but $R$ is only partially observed (95%+ missing) — SVD requires a complete matrix, and naively filling missing entries with 0 or the mean biases the factorization toward those imputed values. Instead, matrix factorization for recommenders minimizes squared error **only over observed entries**: $\min_{P,Q}\sum_{(u,i)\in\text{obs}}(r_{ui}-p_u^\top q_i)^2 + \lambda(\|p_u\|^2+\|q_i\|^2)$. This is non-convex jointly in $(P,Q)$ but convex in each individually, which is exactly what ALS exploits: fix one, solve the other in closed form, alternate.

**Q: Why is item-based collaborative filtering usually preferred over user-based CF in production?**
A: Three reasons: (1) Item similarity is more stable over time than user similarity — a user's taste can shift day to day, but "users who liked The Matrix also liked Inception" is a stable relationship, so item-item similarity matrices can be precomputed and cached with infrequent refresh. (2) The item catalog usually grows far more slowly than the user base, so the $O(|I|^2)$ similarity computation is more tractable than $O(|U|^2)$. (3) Item-based recommendations are naturally explainable ("because you watched X") in a way user-based recommendations aren't. The trade-off is item-based CF still suffers the same cold-start problem for brand-new items with no interaction history — matrix factorization with side features, or hybrid content-based fallbacks, are the usual mitigations (see [04-specialized-domains/05-recommender-systems.md](../04-specialized-domains/05-recommender-systems.md) §8, Cold Start).

**Q: How do you adapt matrix factorization for implicit feedback (clicks/views) instead of explicit ratings?**
A: Implicit feedback has no negative signal — a missing interaction means "unobserved," not "disliked," so treating unobserved pairs as 0 (rating-style) is wrong. The standard fix (Hu-Koren-Volinsky / weighted ALS) reframes the problem: binarize preference $p_{ui} = 1$ if any interaction occurred else 0, and introduce a confidence weight $c_{ui} = 1 + \alpha \cdot r_{ui}$ (r_ui = interaction count/strength) so repeated engagement increases confidence without capping preference at a fixed scale. The loss becomes $\sum_{u,i} c_{ui}(p_{ui}-p_u^\top q_i)^2$, summed over **all** user-item pairs, not just observed ones — unobserved pairs get low confidence (weight ≈1) and are gently pulled toward $p_{ui}=0$ rather than being ignored or treated as hard negatives. This full-matrix sum looks computationally infeasible at scale, but the ALS closed-form update can be restructured to avoid iterating over every item explicitly, using the fact that the confidence matrix is diagonal plus a sparse correction — making WALS tractable for real catalogs.
