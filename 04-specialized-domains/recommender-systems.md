---
module: Specialized Domains
topic: Recommender Systems
subtopic: ""
status: unread
tags: [specializeddomains, ml, recommender-systems]
---
# Recommender Systems

---

## Table of Contents

1. [Why Recommender Systems?](#1-why-recommender-systems)
2. [Types of Recommender Systems](#2-types-of-recommender-systems)
3. [Collaborative Filtering Deep Dive](#3-collaborative-filtering-deep-dive)
4. [Content-Based Filtering](#4-content-based-filtering)
5. [Neural Collaborative Filtering (NCF)](#5-neural-collaborative-filtering-ncf)
6. [Two-Tower Models](#6-two-tower-models)
7. [Learning to Rank](#7-learning-to-rank)
8. [The Cold Start Problem](#8-the-cold-start-problem)
9. [Diversity, Serendipity, and Filter Bubbles](#9-diversity-serendipity-and-filter-bubbles)
10. [Evaluation Metrics](#10-evaluation-metrics)
11. [Production Patterns](#11-production-patterns)
12. [Graph Neural Networks for Recommendations](#12-graph-neural-networks-for-recommendations)
13. [Session-Based Recommendations](#13-session-based-recommendations)
14. [Common Interview Questions with Answers](#14-common-interview-questions-with-answers)

---

## 1. Why Recommender Systems?

**The problem:** Catalogs are vast; human attention is finite. Netflix has 15,000+ titles. Spotify has 100M+ tracks. Amazon has 350M+ products. A user session might touch 20 items. Without a selection mechanism, almost all catalog items are invisible to almost all users, and users are left worse off than if they had a knowledgeable curator.

**The core insight:** User behavior is not random — it clusters. People with similar taste histories make similar choices. Items with similar feature profiles attract similar audiences. These correlations are strong enough that a model trained on past behavior can predict future preferences well enough to beat unguided browsing by a large margin.

**The mechanics:** Given a set of users U, items I, and a sparse matrix of observed interactions R, predict the score r̂(u, i) for all unobserved (u, i) pairs and surface the top-K highest-scoring items per user.

```
R : |U| × |I|  matrix   (very sparse — typically 99%+ empty)
Goal: fill in the blanks in a way that maximizes user satisfaction
```

| Platform | Estimated recommendation value |
|----------|--------------------------------|
| Netflix | $1B/year in retained subscribers |
| Amazon | 35% of revenue from recommendations |
| Spotify | Core to user retention and discovery |
| YouTube | 70% of watch time from recommended content |

**What breaks:** The model optimizes a proxy (past interactions) for a target (user satisfaction). If the proxy is click-through rate, you get clickbait. If it is watch time, you get addictive but regretted consumption. If it is ratings on items already watched, you miss discovery. Getting the objective right is harder than any modeling choice.

---

## 2. Types of Recommender Systems

**The problem:** You need to score items for a user. You have two fundamentally different sources of signal: what other people with similar behavior liked (behavioral signal), and what the item itself is like (content signal). These have complementary failure modes.

**The core insight:** Behavioral signal generalizes well across item types but fails when there is no history. Content signal works from day one for new items but cannot surface items outside the user's known taste profile. Neither alone is sufficient; production systems use both.

### 2.1 Collaborative Filtering (CF)

**The problem:** You have dense interaction history but no item features. How do you generate personalized recommendations?

**The core insight:** Users who behaved similarly in the past will behave similarly in the future. Encode each user by whom they resemble; encode each item by which users liked it. No feature engineering required — all information comes from the interaction graph.

- "Users who bought this also bought..."
- Requires interaction data. Fails for cold users and cold items.

### 2.2 Content-Based Filtering (CB)

**The problem:** New items arrive with no interaction history. CF cannot score them.

**The core insight:** Items can be represented as feature vectors. Users can be represented as preference profiles derived from items they liked. Similarity in feature space predicts compatibility, regardless of whether anyone has interacted with the new item yet.

- Works for new items (just need features). Fails for new users.
- Over-specializes: cannot surface items outside the user's documented taste.

### 2.3 Hybrid Systems

Most production systems are hybrid. Netflix uses:
1. CF to find neighbors
2. Content features for cold start and tie-breaking
3. Contextual signals (time of day, device, recent session)
4. A separate ranking model on top

```
Score(u, i) = α · CF_score(u, i) + β · CB_score(u, i) + γ · Context_score(u, i)
```

The weights α, β, γ can themselves be learned.

### 2.4 Knowledge-Based Systems

Rule-driven, for complex or infrequent purchases (mortgages, cars). More constraint satisfaction than ML. Not covered in depth here.

**What breaks:** Hybrid systems add engineering complexity and require careful feature isolation at serving time. Mixing CF and CB signals without understanding their failure modes leads to systems that fail silently: the CB component masks CF cold-start failures, but you do not know whether the recommendations are actually good.

---

## 3. Collaborative Filtering Deep Dive

> For the classical matrix-factorization math (ALS vs SGD, implicit-feedback WALS) in more derivation-heavy form, see [02-classical-ml/recommender-systems-classical.md](../02-classical-ml/recommender-systems-classical.md) — this section focuses on the neighborhood-CF intuition and its role in the broader system.

### 3.1 User-Based CF

**The problem:** You are a new user on a platform. The platform has no item features — it just knows who watched what. How do you generate a personalized recommendation?

**The core insight:** If user A and user B have highly overlapping watch histories, their future preferences are also likely to overlap. Find the K most similar users (the "neighborhood"), then recommend items those users liked that you have not seen.

**The mechanics:**

Cosine similarity:
```
sim(u, v) = (r_u · r_v) / (||r_u|| · ||r_v||)
```

Pearson correlation (accounts for rating bias — some users rate everything high):
```
sim(u, v) = Σ_i (r_ui - r̄_u)(r_vi - r̄_v) / √[Σ_i(r_ui - r̄_u)² · Σ_i(r_vi - r̄_v)²]
```

Prediction:
```python
def predict_user_based(user_u, item_i, ratings, similarities, K=50):
    neighbors = get_top_K_similar_users(user_u, similarities, K)
    
    numerator = sum(
        similarities[user_u][v] * (ratings[v][item_i] - mean_rating[v])
        for v in neighbors
        if item_i in ratings[v]
    )
    denominator = sum(
        abs(similarities[user_u][v])
        for v in neighbors
        if item_i in ratings[v]
    )
    
    return mean_rating[user_u] + numerator / (denominator + 1e-8)
```

**What breaks:** With 100M users, computing pairwise similarities is O(|U|²) — impossible in real time. User similarity estimates are also noisy because individual users have sparse histories. This led to item-based CF and then matrix factorization.

---

### 3.2 Item-Based CF

**The problem:** User-based CF does not scale. Computing all-pairs user similarity at query time is infeasible, and user profiles are too sparse for reliable similarity estimates.

**The core insight:** Items accumulate far more interactions than individual users, so item-item similarities are more statistically reliable. Crucially, item-item similarities are stable over time ("Inception is similar to The Dark Knight" does not change), so they can be precomputed offline. At query time, you only need to look up the neighbors of items the user already interacted with.

**The mechanics:**

```
score(u, i) = Σ_{j ∈ rated_by_u} sim(i, j) · r(u, j) / Σ_{j} |sim(i, j)|
```

```python
# Offline: precompute item similarities
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# item_matrix: (|I|, |U|) — each row is an item's user interaction vector
item_matrix = ratings.T  # shape: (n_items, n_users)
item_similarities = cosine_similarity(item_matrix)  # (n_items, n_items)

# Online: given user u's history, score all items
def score_items_for_user(user_id, user_history, item_similarities, top_k=10):
    scores = np.zeros(item_similarities.shape[0])
    for item_id, rating in user_history.items():
        scores += item_similarities[item_id] * rating
    for item_id in user_history:
        scores[item_id] = 0
    return np.argsort(scores)[::-1][:top_k]
```

**What breaks:** Item-item CF still fails for new items (no interactions → no neighbors) and for users with very sparse histories. The model also captures only first-order interactions: "people who liked A also liked B." It cannot reason: "people who liked A and B also liked C because A and B share latent property X."

---

### 3.3 Matrix Factorization

**The problem:** Neighborhood methods only use direct co-occurrence. They cannot generalize across items that share latent properties without explicit co-occurrence. Also, both user-based and item-based CF have O(|U|²) or O(|I|²) precomputation costs.

**The core insight:** The interaction matrix has low-rank structure. Most user behavior can be explained by a small number of latent taste dimensions — things like "affinity for action movies," "preference for indie content," "genre diversity." By decomposing R into user factors P (|U| × k) and item factors Q (|I| × k), you learn these dimensions directly from data, without anyone labeling them.

**The mechanics:**

```
R ≈ P · Q^T

P: (|U|, k)  — user latent factors
Q: (|I|, k)  — item latent factors
k: number of latent dimensions (typically 64–512)

r̂(u, i) = p_u · q_i  (dot product)
```

Regularized objective (only over observed entries):
```
min_{P, Q} Σ_{(u,i) observed} (r_ui - p_u^T q_i)² + λ(||p_u||² + ||q_i||²)
```

SGD update:
```python
def sgd_update(p_u, q_i, r_ui, lr=0.01, reg=0.02):
    error = r_ui - np.dot(p_u, q_i)
    p_u_new = p_u + lr * (error * q_i - reg * p_u)
    q_i_new = q_i + lr * (error * p_u - reg * q_i)
    return p_u_new, q_i_new

for epoch in range(n_epochs):
    for u, i, r in observed_ratings:
        p[u], q[i] = sgd_update(p[u], q[i], r)
```

Bias terms (important in practice):
```
r̂(u, i) = μ + b_u + b_i + p_u^T q_i
```

Where μ is the global mean rating, b_u is user bias, b_i is item bias.

**What breaks:** The dot product is linear. It cannot capture non-linear interaction patterns between user and item factors. For implicit feedback (where all unobserved pairs are not true negatives), naive MF also assigns equal confidence to "not interacted with because did not like" and "not interacted with because never saw."

---

### 3.4 ALS (Alternating Least Squares)

**The problem:** SGD for MF requires careful learning rate tuning and sequential updates. It does not parallelize naturally over users and items simultaneously.

**The core insight:** If you fix Q (item factors), the optimization over P becomes a set of independent convex least-squares problems — one per user, each with a closed-form solution. By alternating (fix Q → solve P exactly; fix P → solve Q exactly), both sub-problems are always convex, parallelizable, and have no learning rate to tune.

**The mechanics:**

```
Fix Q → solve for each p_u:
    p_u = (Q^T Q + λI)^{-1} Q^T r_u

Fix P → solve for each q_i:
    q_i = (P^T P + λI)^{-1} P^T r_i
```

```python
from pyspark.ml.recommendation import ALS

als = ALS(
    maxIter=10,
    regParam=0.01,
    userCol="userId",
    itemCol="movieId",
    ratingCol="rating",
    coldStartStrategy="drop",
    implicitPrefs=False  # True for implicit feedback
)

model = als.fit(training_df)
recommendations = model.recommendForAllUsers(10)
```

**What breaks:** ALS requires materializing the full Q^T Q matrix (k × k) per update. For very large k this becomes expensive. Standard ALS also assumes explicit ratings; implicit feedback requires a modified objective (see iALS below).

---

### 3.5 Implicit vs. Explicit Feedback

**The problem:** Most user interactions are implicit — clicks, streams, purchases — not explicit ratings. Implicit data is abundant but noisy: you do not know whether a non-interaction means "did not like" or "never saw." Explicit ratings are sparse (most users never rate anything) and biased (you only rate things you watched, and you watched things you expected to like).

**The core insight:** For implicit data, treat all (user, item) pairs as training examples, but assign each a confidence proportional to the interaction count. High-confidence positives (50 streams) are trustworthy; zero-interaction pairs get low confidence (not trusted negatives, just unconfirmed).

**The mechanics (Hu, Koren, Volinsky 2008 — iALS):**

Confidence:
```
c_ui = 1 + α · f_ui

where f_ui = interaction count (plays, clicks, etc.)
      α = confidence scaling parameter (typically 40)
```

Preference:
```
p_ui = 1 if f_ui > 0 else 0
```

Objective:
```
min_{P, Q} Σ_{u,i} c_ui(p_ui - p_u^T q_i)² + λ(||P||² + ||Q||²)
```

This treats all (user, item) pairs as training examples: zero-interaction pairs get c=1, positive interactions get higher confidence.

**What breaks:** Even with confidence weighting, very popular items get higher cumulative confidence from multiple low-quality interactions. The model still cannot distinguish "did not know this existed" from "saw it and chose not to engage."

---

## 4. Content-Based Filtering

### 4.1 The Core Idea

**The problem:** New items have no interaction history. CF cannot score them. You need a way to recommend items on the basis of what they are, not who has interacted with them.

**The core insight:** Items can be represented as feature vectors. Users can be represented as preference profiles (derived from the average features of items they liked). Similarity in feature space predicts compatibility.

### 4.2 TF-IDF for Text-Based Items

**The problem:** A movie's description, a news article's body, a product's title — these are bags of words, but not all words are equally informative. "The" appears in every document and carries no signal. "Heist" appears rarely and is highly discriminative.

**The core insight:** Weight each word by how often it appears in this document (TF) and how rare it is across all documents (IDF). Words that appear often in one document but rarely elsewhere are the most informative.

**The mechanics:**

```
TF(w, d) = count(w in d) / total_words(d)
IDF(w) = log(N / df(w))     [N = total docs, df(w) = docs containing w]
TF-IDF(w, d) = TF(w, d) · IDF(w)
```

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

movies = pd.DataFrame({
    'title': ['The Dark Knight', 'Inception', 'Interstellar', 'The Notebook'],
    'description': [
        'Batman fights Joker in Gotham City crime thriller',
        'Dream heist sci-fi thriller mind-bending',
        'Space time wormhole family sci-fi',
        'Romance love story emotional drama'
    ]
})

vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
item_matrix = vectorizer.fit_transform(movies['description'])

item_sim = cosine_similarity(item_matrix)

def build_user_profile(liked_item_indices, item_matrix):
    liked_vecs = item_matrix[liked_item_indices]
    return liked_vecs.mean(axis=0)

def score_items(user_profile, item_matrix, already_seen):
    scores = cosine_similarity(user_profile, item_matrix).flatten()
    scores[already_seen] = 0
    return scores.argsort()[::-1][:10]
```

**What breaks:** TF-IDF ignores word order and semantics. "Bank of a river" and "bank robbery" both contain "bank" — TF-IDF treats them identically. Use sentence embeddings (e.g., SBERT) when semantic similarity matters more than keyword overlap.

### 4.3 Feature Engineering for Different Domains

**Movies/TV:** Genre (one-hot), cast/director (entity embeddings), plot synopsis (TF-IDF or sentence embeddings), release year, MPAA rating.

**Music (Spotify):** Audio features — tempo, energy, danceability, valence, loudness; genre tags; artist embeddings; lyrics (NLP features).

**E-commerce (Amazon):** Category hierarchy, brand, price tier, product attributes (size, color, material), image embeddings (visual similarity).

**The representation bottleneck:** All these heterogeneous features must map into a unified embedding space for comparison. This is why neural methods (Sections 5, 6) took over — they learn the feature mapping jointly with the recommendation task.

### 4.4 Content-Based Pros and Cons

| Pros | Cons |
|------|------|
| No cold start for new items | Cold start for new users |
| Explainable ("because you liked action movies") | Feature engineering burden |
| No popularity bias | Cannot discover items outside user's known taste |
| Works for niche tastes | Overspecialization — no serendipity |

**What breaks:** Pure content-based filtering creates a taste bubble. A user who liked two action movies will only ever see more action movies. It also requires high-quality item features, which are expensive to engineer and maintain.

---

## 5. Neural Collaborative Filtering (NCF)

**The problem:** Matrix factorization uses a dot product to combine user and item embeddings. The dot product is linear. It cannot capture complex, non-linear interactions: "user likes fast-paced action" and "item is a slow psychological thriller" might still be a good match in ways a linear function cannot represent.

**The core insight:** Replace the dot product with a neural network. Feed the concatenated user and item embeddings into an MLP that can learn arbitrary non-linear compatibility functions. You lose the geometric interpretability of MF but gain representational power.

**The mechanics:**

```
User ID ─→ User Embedding ─┐
                           ├─→ [Concatenate] ─→ MLP layers ─→ r̂(u,i)
Item ID ─→ Item Embedding ─┘
```

```python
import torch
import torch.nn as nn

class NCF(nn.Module):
    def __init__(self, n_users, n_items, embed_dim=64, mlp_layers=[128, 64, 32]):
        super().__init__()
        
        # GMF branch (Generalized Matrix Factorization)
        self.gmf_user_embed = nn.Embedding(n_users, embed_dim)
        self.gmf_item_embed = nn.Embedding(n_items, embed_dim)
        
        # MLP branch
        self.mlp_user_embed = nn.Embedding(n_users, embed_dim)
        self.mlp_item_embed = nn.Embedding(n_items, embed_dim)
        
        mlp_input_dim = embed_dim * 2
        layers = []
        for out_dim in mlp_layers:
            layers.extend([nn.Linear(mlp_input_dim, out_dim), nn.ReLU()])
            mlp_input_dim = out_dim
        self.mlp = nn.Sequential(*layers)
        
        self.predict = nn.Linear(embed_dim + mlp_layers[-1], 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, user_ids, item_ids):
        gmf_user = self.gmf_user_embed(user_ids)
        gmf_item = self.gmf_item_embed(item_ids)
        gmf_out = gmf_user * gmf_item  # element-wise product
        
        mlp_user = self.mlp_user_embed(user_ids)
        mlp_item = self.mlp_item_embed(item_ids)
        mlp_input = torch.cat([mlp_user, mlp_item], dim=-1)
        mlp_out = self.mlp(mlp_input)
        
        combined = torch.cat([gmf_out, mlp_out], dim=-1)
        output = self.sigmoid(self.predict(combined))
        return output.squeeze()
```

Training uses negative sampling: for each positive (u, i) pair, sample n_neg random items the user has not interacted with.

```python
def sample_negatives(pos_user_item_pairs, n_items, n_neg=4):
    negatives = []
    user_history = build_user_history(pos_user_item_pairs)
    
    for user, pos_item in pos_user_item_pairs:
        count = 0
        while count < n_neg:
            neg_item = random.randint(0, n_items - 1)
            if neg_item not in user_history[user]:
                negatives.append((user, neg_item, 0))
                count += 1
    
    return negatives

criterion = nn.BCELoss()
```

**What breaks:** NCF requires a separate forward pass for each (user, item) pair at scoring time. With 100M items, that is 100M forward passes per user query — infeasible for retrieval. NCF works well as a re-ranker over a small candidate set, not as a retrieval model over the full catalog.

---

## 6. Two-Tower Models

### 6.1 The Scale Problem

**The problem:** Any model that requires joint processing of user and item features (NCF, MF with concatenation) cannot scale to a catalog of 100M+ items. You cannot run a forward pass for every item at every user request.

**The core insight:** Separate the user and item computation completely. If the user embedding and item embedding only interact via a final dot product, you can precompute all item embeddings offline and use approximate nearest neighbor (ANN) search to retrieve top-K items in milliseconds. The entire catalog search collapses to one user forward pass plus one ANN lookup.

**The mechanics:**

```
User features ─→ User Tower (MLP) ─→ user_embedding (d-dim)
                                              ↘
                                               dot product ─→ score
                                              ↗
Item features ─→ Item Tower (MLP) ─→ item_embedding (d-dim)
```

```python
class TwoTowerModel(nn.Module):
    def __init__(self, user_feature_dim, item_feature_dim, embed_dim=128):
        super().__init__()
        
        self.user_tower = nn.Sequential(
            nn.Linear(user_feature_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
        self.item_tower = nn.Sequential(
            nn.Linear(item_feature_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, embed_dim),
            nn.LayerNorm(embed_dim)
        )
    
    def forward(self, user_features, item_features):
        user_emb = self.user_tower(user_features)
        item_emb = self.item_tower(item_features)
        scores = torch.sum(user_emb * item_emb, dim=-1)
        return scores
    
    def get_user_embedding(self, user_features):
        return self.user_tower(user_features)
    
    def get_item_embedding(self, item_features):
        return self.item_tower(item_features)
```

### 6.2 Training

**The problem:** For each (user, positive_item) pair, you need negative items to contrast against. Sampling negatives separately is expensive and biased toward the uniform distribution over items.

**The core insight:** For a batch of B (user, positive_item) pairs, treat every other item in the batch as a negative for each user. This gives B-1 negatives "for free" — the distribution of in-batch negatives is roughly proportional to item popularity, which is closer to the true marginal than uniform sampling.

**In-batch negatives:**
```python
def two_tower_loss(user_embs, item_embs, temperature=0.07):
    """
    user_embs: (B, d)
    item_embs: (B, d) — one positive item per user
    """
    logits = torch.matmul(user_embs, item_embs.T) / temperature
    labels = torch.arange(logits.shape[0]).to(logits.device)
    loss = nn.CrossEntropyLoss()(logits, labels)
    return loss
```

**Sampling bias correction:** Popular items appear more often in batches as negatives, causing the model to over-penalize popular items. Fix: subtract log(p_i) from item scores where p_i is the item's sampling probability.

```
corrected_score(u, i) = score(u, i) - log(p_i)
```

### 6.3 Retrieval with ANN

```python
import faiss
import numpy as np

d = 128
item_embeddings = get_all_item_embeddings()  # (n_items, d)

index = faiss.IndexFlatIP(d)  # Inner product (cosine if normalized)
index.add(item_embeddings.astype(np.float32))

def retrieve_candidates(user_features, k=500):
    user_emb = model.get_user_embedding(user_features)
    user_emb = user_emb.detach().numpy().astype(np.float32)
    distances, indices = index.search(user_emb.reshape(1, -1), k)
    return indices[0], distances[0]
```

### 6.4 Two-Stage Architecture

The two-tower retrieves candidates (top 500-1000). A separate, more expensive ranking model then scores those candidates with richer features.

```
All items (100M)
     ↓ [Two-Tower ANN retrieval]
Candidate set (~500 items)
     ↓ [Ranking model (DNN, LightGBM, etc.)]
Ranked list (top 50)
     ↓ [Re-ranking for diversity, business rules]
Final slate shown to user
```

### 6.5 YouTube Deep Neural Net (Covington et al., 2016)

The canonical two-tower paper. Key design choices:
- **Candidate generation:** User tower takes watch history (average of video embeddings), search tokens, demographics, context. Item tower: video features.
- **Example age feature:** Time since video was uploaded — prevents the model from learning "old videos are good because we have observed them long enough."
- **Asymmetric co-watch:** Training treats the most recently watched video as the label, with prior watches as context — predicts "what will you watch next."

**What breaks:** The two-tower's dot product interaction is less expressive than NCF's MLP. User-item feature crosses (e.g., "user is in age group X AND item is genre Y") cannot be captured within the towers. This is why two-towers are used for retrieval (must be fast, moderate quality) and a richer model is used for ranking (can be slow, must be high quality).

---

## 7. Learning to Rank

### 7.1 Why Ranking Matters

**The problem:** Retrieval gives you 500 candidates. They need to be sorted. The first result gets 10x more clicks than the fifth. Pointwise scoring — predicting an absolute relevance score for each item independently — does not optimize what users see at the top of the list.

**The core insight:** What matters is relative order, not absolute scores. Models should be trained to compare items against each other, with the optimization weighted by how much a position swap would affect user experience.

### 7.2 Three Paradigms

#### 7.2.1 Pointwise

**The problem:** You need a relevance score. The simplest approach is regression or classification on each item independently.

**The core insight:** Predict the relevance of each (user, item) pair as an absolute score, then rank by score. Simple but does not model the comparisons that determine rank order.

```
Loss = Σ_{u,i} (r̂(u,i) - r(u,i))²  [MSE for ratings]
```

**What breaks:** Predicting absolute relevance scores independently does not optimize order. You can predict all scores perfectly wrong (systematically off by a constant) and still have correct rankings, or get the wrong rankings with low MSE.

#### 7.2.2 Pairwise

**The problem:** Pointwise methods do not model the comparisons that determine rank order.

**The core insight:** For each user, create pairs (i, j) where i is more relevant than j. Minimize the probability of ranking j above i. This directly models the comparisons that determine the final list.

**BPR (Bayesian Personalized Ranking):** For user u, positive item i, negative item j:

```
L_BPR = -Σ_{(u,i,j)} log σ(r̂_ui - r̂_uj) + λ||Θ||²
```

```python
def bpr_loss(pos_scores, neg_scores, reg_lambda=0.01):
    diff = pos_scores - neg_scores
    loss = -torch.log(torch.sigmoid(diff)).mean()
    return loss
```

**What breaks:** Pairwise methods treat each pair independently but do not account for position — swapping items at rank 1 and 2 matters far more than swapping items at rank 50 and 51.

#### 7.2.3 Listwise

**The problem:** Pairwise methods do not account for position — swapping items at rank 1 and 2 matters far more than swapping items at rank 50 and 51.

**The core insight:** Define the optimization directly in terms of the metric you care about (NDCG). Even if NDCG is not differentiable, you can define what the gradient should be as a function of how much each swap would change NDCG.

**LambdaRank / LambdaMART:**

```
λ_ij = ∂L/∂(s_i - s_j) = -σ / (1 + exp(σ(s_i - s_j))) · |ΔNDCG|
```

The |ΔNDCG| term weights the gradient by "how much would this swap improve the ranking?" Large rank swaps near the top of the list get large gradients.

**LambdaMART** = LambdaRank + MART (gradient boosted trees). The standard workhorse at Microsoft/Bing and many production ranking systems.

```python
import lightgbm as lgb

params = {
    'objective': 'lambdarank',
    'metric': 'ndcg',
    'ndcg_eval_at': [1, 3, 5, 10],
    'num_leaves': 31,
    'min_data_in_leaf': 20,
    'num_iterations': 500,
    'learning_rate': 0.05,
    'label_gain': [0, 1, 3, 7, 15]
}

train_data = lgb.Dataset(X_train, label=y_train, group=group_train)
model = lgb.train(params, train_data, valid_sets=[valid_data])
```

### 7.3 Summary Comparison

| Approach | Optimizes | Typical Use | Pros | Cons |
|----------|-----------|-------------|------|------|
| Pointwise | Individual relevance | Rating prediction | Simple | Ignores order |
| Pairwise (BPR) | Relative order | Top-K ranking | Better than pointwise | Does not optimize full list quality |
| Listwise (LambdaMART) | List quality (NDCG) | Production ranking | Best metrics | Slower, harder to implement |

**What breaks:** Listwise methods still optimize offline proxies. NDCG on held-out interactions may not correlate with the business metric you actually care about (watch time, conversion). The relationship between offline NDCG and online CTR is empirically weak. Always validate with A/B tests.

---

## 8. The Cold Start Problem

**The problem:** Collaborative filtering needs interaction history to make predictions. New users have none. New items have none. The model cannot score them. This is not a corner case — every user starts cold, and platforms continuously add new items.

**The core insight:** Cold start is not one problem but two, and they have different solutions. New users need preference elicitation strategies (questionnaires, exploration slates). New items need feature-based representations that work without interaction history.

### 8.1 New User Strategies

**Onboarding questionnaires:** Spotify's "pick 3 artists you like." Explicit preference elicitation. Fast, works, but users resist long surveys.

**Demographic-based:** Use age, location, device type to find similar users who have been around longer.

**Popular items fallback:** Recommend the most popular items in broad categories. Safe but not personalized.

**Exploration slate:** First session shows a diverse set designed to maximize information gain about user taste, not maximize immediate click rate.

```python
def cold_start_explore_slate(user, session_history, all_items, n=20):
    """
    For a cold-start user, recommend a diverse slate that maximizes 
    information gain about their taste.
    """
    cluster_centers = get_cluster_centers(all_items)
    
    candidates = []
    for cluster_id in range(n):
        top_item = get_top_item_in_cluster(cluster_id, min_popularity=100)
        candidates.append(top_item)
    
    return candidates[:n]
```

### 8.2 New Item Strategies

**Content-based features:** New item? Extract features and find similar established items. Seed recommendations with those items' user bases.

```
new_item_embedding ≈ average(embeddings of k most similar existing items)
```

**Warm-up injection:** Reserve 1 slot in 20 for new item exploration. Collect interaction data. Graduate to full CF once enough data accumulates.

**Feature-based item tower:** Train item embeddings that can be computed from features alone, not just item ID. New item features → immediate embedding → ANN retrieval works on day 1.

```python
class FeatureBasedItemEncoder(nn.Module):
    """
    Item tower that accepts features instead of IDs.
    Allows zero-shot embedding for new items.
    """
    def __init__(self, feature_dim, embed_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, embed_dim),
            nn.LayerNorm(embed_dim)
        )
    
    def forward(self, item_features):
        return self.encoder(item_features)
```

### 8.3 Meta-Learning Approaches

**MAML / FOMAML for cold start:** Frame as few-shot learning. Train the model to quickly adapt to a new user/item from a small number of examples. Practical in production at large companies.

**What breaks:** Exploration slates hurt short-term engagement metrics (users do not always like being shown unfamiliar content). New item warm-up requires explicitly reserving slots, which means lower short-term CTR for those slots. These are acceptable costs; systems that do not pay them become stale.

---

## 9. Diversity, Serendipity, and Filter Bubbles

**The problem:** Pure accuracy optimization creates a feedback loop. User watches sci-fi → more sci-fi recommended → user watches those → even more sci-fi recommended → user's profile collapses to a narrow cluster → user stops discovering new genres → user disengages and churns. Optimizing immediate click probability is not the same as optimizing long-term retention.

**The core insight:** The recommendation slate is not just a list of items — it is a portfolio. A portfolio should cover multiple preferences to maximize expected satisfaction across the session and across time. Diversity and calibration are objectives to be optimized alongside relevance.

### 9.1 Diversity

**Intra-list diversity (ILD):** The slate of K items should cover different taste dimensions.

```
ILD(L) = (2 / (|L|(|L|-1))) · Σ_{i,j ∈ L, i≠j} (1 - sim(i, j))
```

**MMR (Maximal Marginal Relevance):** Greedily select items that balance relevance and diversity.

```python
def mmr_reranking(candidates, user_embedding, item_embeddings, 
                   already_selected=[], lambda_=0.5, k=10):
    selected = list(already_selected)
    remaining = list(candidates)
    
    for _ in range(k):
        best_score = -float('inf')
        best_item = None
        
        for item in remaining:
            relevance = cosine_similarity(
                user_embedding, item_embeddings[item]
            )
            
            if selected:
                redundancy = max(
                    cosine_similarity(item_embeddings[item], item_embeddings[s])
                    for s in selected
                )
            else:
                redundancy = 0
            
            mmr_score = lambda_ * relevance - (1 - lambda_) * redundancy
            
            if mmr_score > best_score:
                best_score = mmr_score
                best_item = item
        
        selected.append(best_item)
        remaining.remove(best_item)
    
    return selected
```

### 9.2 Serendipity

Serendipity = unexpected but pleasant. Not just items outside the user's current profile, but items outside the profile that the user ends up enjoying. Measuring serendipity is hard — it requires observing post-interaction satisfaction, not just whether the item was clicked.

Proxy: items that are dissimilar to the user's current profile but end up receiving high ratings or high completion rates.

### 9.3 Filter Bubbles

The societal concern: recommendation algorithms may reinforce existing beliefs and preferences by only showing users content that confirms what they already think.

Mitigation:
- **Exploration budget:** Reserve a fixed fraction of slots for out-of-distribution recommendations.
- **Diversity constraints:** Hard constraints on genre/topic distribution per slate.
- **Temporal freshness:** Ensure recently published content is surfaced.
- **Serendipity objectives:** Add serendipity as a term in the optimization objective.

### 9.4 Calibration

**The problem:** A user who watches 60% action movies and 40% documentaries but only receives action movie recommendations is being served a miscalibrated slate. The model is over-representing one part of the user's taste.

**The core insight:** The recommendation distribution should reflect the user's historical consumption distribution across categories. Divergence from this target is measurable and minimizable.

```
Calibration loss = KL divergence(target distribution, recommendation distribution)
```

Where target distribution comes from the user's historical consumption across categories.

**What breaks:** Diversity and calibration are harder to optimize than relevance because they are set-level properties, not item-level. They require re-ranking algorithms (MMR, DPP) that are computationally more expensive than individual scoring. The tradeoff parameter λ is tuned via A/B testing, not offline metrics.

---

## 10. Evaluation Metrics

### 10.1 The Problem with MSE/MAE

**The problem:** Rating prediction (minimizing MSE on a held-out test set) does not measure what users experience. A system that predicts 4.1 stars vs. 4.2 stars for two items in the wrong order has low MSE but is actually wrong about what to show first. Users only see the top K items; quality at rank 1000 is irrelevant.

**The core insight:** Evaluation must measure ranking quality, especially at the top of the list. A metric that rewards getting the right items at the right positions, and penalizes missed relevant items, is closer to actual user experience than MSE.

### 10.2 Precision@K

**The problem:** Of the K items recommended, what fraction are relevant?

**The core insight:** Measures recommendation precision but ignores how many relevant items you missed.

```
Precision@K = |{relevant items} ∩ {top K recommended}| / K
```

```python
def precision_at_k(recommended, relevant, k):
    recommended_k = recommended[:k]
    hits = len(set(recommended_k) & set(relevant))
    return hits / k
```

**What breaks:** Does not penalize a system for missing relevant items. If you have 100 relevant items and return 10 all correct, Precision@10 = 1.0 — but you missed 90.

### 10.3 Recall@K

**The problem:** Precision does not tell you what fraction of all relevant items you surfaced.

**The core insight:** Measures coverage of the relevant set, but treats all positions within top K equally.

```
Recall@K = |{relevant items} ∩ {top K recommended}| / |{relevant items}|
```

```python
def recall_at_k(recommended, relevant, k):
    recommended_k = recommended[:k]
    hits = len(set(recommended_k) & set(relevant))
    return hits / max(len(relevant), 1)
```

**What breaks:** Does not account for rank order within the top K. A relevant item at rank 1 and at rank K both contribute equally.

### 10.4 NDCG (Normalized Discounted Cumulative Gain)

**The problem:** Precision and Recall treat all positions within the top K equally. But position 1 matters far more than position K.

**The core insight:** Discount the relevance of items by their position using a logarithmic decay. Normalize by the ideal ranking so scores are comparable across users with different numbers of relevant items.

**DCG@K:**
```
DCG@K = Σ_{i=1}^{K} (2^{rel_i} - 1) / log_2(i + 1)
```

Relevance at position 1 is weighted at 1/log(2) = 1.0; position 2 at 1/log(3) ≈ 0.63; position 3 at 0.5; etc.

**NDCG@K = DCG@K / IDCG@K** — normalized to [0, 1] where IDCG is the DCG of the ideal ranking.

```python
import numpy as np

def dcg_at_k(relevances, k):
    relevances = np.array(relevances[:k])
    if len(relevances) == 0:
        return 0
    discounts = np.log2(np.arange(2, len(relevances) + 2))
    return np.sum((2**relevances - 1) / discounts)

def ndcg_at_k(recommended, relevant_with_scores, k):
    gains = [relevant_with_scores.get(item, 0) for item in recommended[:k]]
    ideal_gains = sorted(relevant_with_scores.values(), reverse=True)[:k]
    dcg = dcg_at_k(gains, k)
    idcg = dcg_at_k(ideal_gains, k)
    return dcg / (idcg + 1e-8)
```

**What breaks:** NDCG assumes you know the relevance scores for all items — in practice, you only know about items the user interacted with. Items the user would have liked but never saw are invisible to the metric, causing systematic underestimation of a retrieval model's quality gap.

### 10.5 MAP (Mean Average Precision)

Average Precision for one user, averaged over all users.

```
AP = (1/|relevant|) · Σ_{k=1}^{K} Precision@k · rel(k)
```

where rel(k) = 1 if item at rank k is relevant, else 0.

```python
def average_precision(recommended, relevant):
    hits = 0
    sum_precision = 0
    
    for k, item in enumerate(recommended, 1):
        if item in relevant:
            hits += 1
            sum_precision += hits / k
    
    return sum_precision / max(len(relevant), 1)

def mean_average_precision(all_recommended, all_relevant):
    return np.mean([
        average_precision(rec, rel) 
        for rec, rel in zip(all_recommended, all_relevant)
    ])
```

### 10.6 Hit Rate@K (HR@K)

Binary: did the user interact with at least one recommended item?

```
HR@K = (1/|U|) · Σ_u 1[relevant ∩ top_K(u) ≠ ∅]
```

Simple and easy to interpret. Common in industry for sparse interaction data where even one hit is a success.

### 10.7 MRR (Mean Reciprocal Rank)

Where does the first relevant item appear?

```
MRR = (1/|U|) · Σ_u (1 / rank_of_first_relevant_item(u))
```

Best for "did the right answer appear near the top?" scenarios. Less appropriate when there are many relevant items.

### 10.8 Coverage and Catalog Utilization

```
Catalog coverage = |unique items recommended| / |total items|
User coverage = |users receiving personalized recs| / |total users|
```

### 10.9 Business Metrics

All the above are offline proxy metrics. What production actually optimizes:

| Business Metric | Description |
|-----------------|-------------|
| CTR (Click-Through Rate) | Fraction of recommended items clicked |
| CVR (Conversion Rate) | Fraction of recommendations leading to purchase |
| Watch time / listen time | Engagement depth, not just click |
| DAU/MAU | Daily/Monthly active users |
| Retention | 30-day return rate |
| LTV (Lifetime Value) | Long-term user value |

**Warning:** Optimizing CTR can hurt long-term retention (clickbait). Most sophisticated systems optimize a blended objective.

### 10.10 Offline vs. Online Evaluation

| | Offline | Online (A/B) |
|---|---------|--------------|
| Speed | Hours/days | Weeks |
| Cost | Cheap | Expensive (real users) |
| Measures | Proxy metrics | Real business metrics |
| Ground truth | Historical data | Fresh user behavior |
| Recommendation | Development/filtering | Launch decision |

**What breaks:** Offline NDCG improvements do not reliably translate to online CTR improvements. The correlation between offline and online metrics is empirically weak. Always validate promising offline results with A/B tests.

---

## 11. Production Patterns

### 11.1 The Recommendation Pipeline

**The problem:** No single model can do everything well. A model that is accurate enough to rank well also takes too long to score 100M items. A model that is fast enough for retrieval lacks the feature richness for precise ranking.

**The core insight:** Split the problem into stages with different quality/speed tradeoffs. Retrieval must be fast (milliseconds over 100M items) but only needs to recall the right items into a candidate set. Ranking can be slow (it sees only 500 items) but must precisely order them.

```
User request arrives
        ↓
[Retrieval Layer]
  - ANN lookup (Two-Tower)
  - BM25/fuzzy matching
  - Popular items fallback
  → 200-1000 candidates
        ↓
[Feature Assembly]
  - Fetch user features (profile, recent history)
  - Fetch item features (metadata, stats)
  - Compute cross features
        ↓
[Ranking Layer]
  - DNN / LightGBM ranking model
  - Scores all 200-1000 candidates
  → Top 50
        ↓
[Re-ranking Layer]
  - Diversity constraints (MMR)
  - Business rules (sponsored items, fresh content)
  - Context injection (trending, seasonal)
  → Final 10-20 items
        ↓
[Response]
```

### 11.2 Real-Time vs. Batch Scoring

**Batch (offline) scoring:** Precompute recommendations for all users nightly. Fast serving; stale. Good for "weekly playlist" style features.

**Near-real-time (streaming):** Update user embeddings as events stream in (Kafka + Flink). Fast retrieval. Stale within a session.

**Real-time scoring:** Full ranking model inference at request time. Higher latency (typically 50-200ms budget for the entire pipeline). Most production systems use real-time retrieval + real-time ranking.

**The latency budget (typical e-commerce):**
```
Total budget:        100ms
├── Feature fetch:    20ms  (Redis/Memcached)
├── ANN retrieval:    10ms  (FAISS / ScaNN)
├── Feature assembly: 15ms
├── Ranking inference: 30ms (model server)
└── Serialization:    10ms
                      85ms  (15ms buffer)
```

### 11.3 Feature Stores

**The problem:** Features must be available at both training time and serving time, with identical values. Without a shared store, training-serving skew is inevitable: features computed differently at training and inference time mean the model is trained on data it will never see in production.

**The core insight:** A feature store decouples feature computation from feature consumption. Write feature transformations once; they run identically at training time (batch) and serving time (real-time lookup).

```
Data sources → Feature Store → Training pipeline
                     ↓
             Serving layer → Real-time inference
```

Classic bugs that feature stores prevent:
- Time leakage: using future information at training time
- Different normalization between training and serving
- Missing values handled differently

### 11.4 Caching

Item embeddings: precomputed, cached in memory (FAISS index). Refresh daily or weekly.

User embeddings: more dynamic.
1. Precompute hourly (batch) — stale but fast
2. Compute on-demand with caching (1-hour TTL)
3. Streaming update as events arrive

Recommendation results: cache per (user_id, context) with short TTL (5-15 minutes). Must invalidate on significant new interactions.

### 11.5 A/B Testing

**Basics:** Split users randomly into control (A) and treatment (B). Measure business metrics over 2-4 weeks. Decide based on statistical significance.

**Common pitfalls:**
- **Network effects:** If users interact with each other, A/B contamination occurs (social platforms).
- **Novelty effect:** New recommendations get inflated engagement just because they are new (usually fades in 2-4 weeks).
- **Multiple testing:** Running 10 experiments simultaneously inflates false positive rate → use Bonferroni correction or sequential testing.
- **Carryover effects:** User behavior changes persist after the experiment ends.

**Interleaving:** Interleave A and B results in a single ranked list and measure which items get clicked. Faster results (more sensitive), no user-split contamination risk. Used heavily at Netflix.

### 11.6 Monitoring

```python
# Coverage — are we recommending diverse items?
coverage = len(unique_recommended_items_today) / total_catalog_size

# Popularity bias — are we over-recommending popular items?
popularity_of_recommended = mean(item_popularity for item in recommendations)

# Freshness — are we surfacing new content?
age_of_recommended = mean(days_since_published for item in recommendations)
```

**Drift detection:** Trigger retraining when CTR drops more than 5% vs. 7-day moving average, NDCG on held-out set drops below threshold, or feature distribution shifts significantly (PSI/KL divergence monitoring).

**What breaks:** The pipeline adds latency at every stage. Feature assembly is often the bottleneck, not model inference. Caching reduces latency but creates consistency problems: a user who just watched something should not be recommended the same thing in the next 5 minutes, even if their cached user embedding has not been refreshed.

---

## 12. Graph Neural Networks for Recommendations

**The problem:** Matrix factorization and two-tower models capture only first-order user-item interactions. They cannot model the signal buried in multi-hop paths: "User A liked Item X → Item X was also liked by User B → User B liked Item Y → recommend Item Y to User A." This second-order collaborative signal is real and captures important taste patterns.

**The core insight:** The user-item interaction matrix is a bipartite graph. Multi-hop reasoning over this graph is equivalent to neighborhood aggregation in a GNN. Graph convolution makes higher-order collaborative filtering explicit and learnable.

### 12.1 Graph Construction

```
Bipartite graph G = (U ∪ I, E)
- Nodes: users + items
- Edges: user-item interactions (weighted by interaction strength)

Can also be augmented:
- Item-item edges (same category, bought together)
- User-user edges (social connections)
- Attribute nodes (genre, brand)
```

### 12.2 LightGCN (He et al., 2020)

**The problem:** Standard GCN applies feature transformation and non-linear activation at each layer. For recommendation, where the "node features" are just trainable embeddings, these transformations add parameters without benefit and make training harder.

**The core insight:** Remove feature transformation and non-linear activation entirely. Just propagate embeddings over the graph. The only trainable parameters are the initial embeddings; all generalization comes from graph structure.

**The mechanics:**

```
e_u^(0) = user embedding (trainable)
e_i^(0) = item embedding (trainable)

Propagation rule (layer k):
e_u^(k) = Σ_{i ∈ N(u)} (1/√|N(u)||N(i)|) · e_i^(k-1)
e_i^(k) = Σ_{u ∈ N(i)} (1/√|N(i)||N(u)|) · e_u^(k-1)

Final embedding: average across layers
e_u = (1/K) Σ_{k=0}^{K} e_u^(k)
```

```python
class LightGCN(nn.Module):
    def __init__(self, n_users, n_items, embed_dim=64, n_layers=3):
        super().__init__()
        self.n_layers = n_layers
        
        self.user_embed = nn.Embedding(n_users, embed_dim)
        self.item_embed = nn.Embedding(n_items, embed_dim)
        nn.init.normal_(self.user_embed.weight, std=0.01)
        nn.init.normal_(self.item_embed.weight, std=0.01)
    
    def forward(self, adj_matrix):
        all_user_embs = [self.user_embed.weight]
        all_item_embs = [self.item_embed.weight]
        
        user_emb = self.user_embed.weight
        item_emb = self.item_embed.weight
        
        for _ in range(self.n_layers):
            new_user_emb = torch.sparse.mm(adj_matrix['UI'], item_emb)
            new_item_emb = torch.sparse.mm(adj_matrix['IU'], user_emb)
            user_emb = new_user_emb
            item_emb = new_item_emb
            all_user_embs.append(user_emb)
            all_item_embs.append(item_emb)
        
        final_user = torch.stack(all_user_embs, dim=1).mean(dim=1)
        final_item = torch.stack(all_item_embs, dim=1).mean(dim=1)
        
        return final_user, final_item
    
    def predict(self, user_ids, item_ids, adj_matrix):
        user_embs, item_embs = self.forward(adj_matrix)
        return torch.sum(user_embs[user_ids] * item_embs[item_ids], dim=-1)
```

**What breaks:** Over-smoothing at deep layers makes all user and item embeddings converge to similar values — the same failure mode as in standard GNNs. More than 3-4 layers typically hurts performance.

### 12.3 PinSage (Pinterest)

**The problem:** The full bipartite graph for a billion-scale RS does not fit in memory. Standard GNN training requires the entire adjacency matrix for each forward pass.

**The core insight:** Sample local neighborhoods via random walks rather than using the full graph. Weight neighbor contributions by how often they appeared in random walks (importance pooling) — this approximates the full graph aggregation while fitting in memory.

Pinterest's GNN for billion-scale recommendations. Key innovations:
1. **Random walk sampling:** Full graph does not fit in memory. Sample local neighborhoods via random walks.
2. **Importance pooling:** Weight neighbor contributions by how often they appeared in random walks.
3. **Curriculum training:** Start with easy negatives, graduate to harder ones.
4. **MapReduce inference:** Compute embeddings in parallel across the graph.

### 12.4 Knowledge Graphs

Augment the bipartite user-item graph with a knowledge graph (item attributes, entity relationships):

```
Item "Inception" → directed_by → "Christopher Nolan"
"Christopher Nolan" → also_directed → "The Dark Knight"
"The Dark Knight" → genre → "Action"
```

Systems like KGCN, KGNN-LS, and RippleNet propagate user preferences through the KG, enabling more interpretable and generalizable recommendations.

### 12.5 When to Use GNNs vs. Standard CF

| | Standard CF | GNN |
|---|-------------|-----|
| Data scale | Works well | Scales with tricks (PinSage) |
| Multi-hop reasoning | Implicit (limited) | Explicit, controllable |
| Side information | Hard to incorporate | Natural (add attribute nodes) |
| Interpretability | Low | Can trace graph paths |
| Engineering complexity | Lower | Higher |
| Best for | Dense interactions | Sparse + rich graph structure |

**What breaks:** GNNs for RS inherit all GNN failure modes. Over-smoothing at deep layers makes user and item embeddings indistinguishable. The full bipartite graph for 100M users × 10M items does not fit in memory — requiring neighbor sampling, which introduces variance. The adjacency matrix changes every day as users interact, requiring expensive index updates.

---

## 13. Session-Based Recommendations

**The problem:** Long-term user history may be irrelevant to the current moment. A user who has watched 500 action movies over 3 years is currently browsing gift ideas for a child. Their session context — the last 10 items viewed — is a better signal than their lifetime profile. Also, anonymous users have no history at all.

**The core insight:** Within a session, item transitions are sequential and Markovian in structure. A model that reads the session as a sequence — predicting what comes next given what came before — can capture the user's current intent without any knowledge of their long-term profile.

### 13.1 Markov Chain Models

**The problem:** You need a baseline that captures sequential dependencies in session data without any training.

**The core insight:** Assume the next item depends only on the last item (first-order Markov). Estimate transition probabilities from co-occurrence counts.

```
P(next item = j | last item = i) = count(i→j) / count(i)
```

Easy to implement, interpretable, but ignores non-adjacent items in the session.

**What breaks:** First-order Markov ignores all items in the session except the most recent one. A user who viewed "running shoes → running socks → GPS watches" has strong evidence of interest in running gear — but first-order MC only knows they viewed GPS watches.

### 13.2 RNN-Based Session Models

**The problem:** Markov chains only use the last item. But the full session context matters: "user viewed running shoes, then running socks, then GPS watches" is more informative than just "user last viewed GPS watches."

**The core insight:** Treat the session as a sequence and encode it with a recurrent network. The hidden state accumulates context from the entire session without requiring explicit transition probability tables.

**GRU4Rec (Hidasi et al., 2015):**

```python
class GRU4Rec(nn.Module):
    def __init__(self, n_items, embed_dim=128, hidden_dim=256, n_layers=1):
        super().__init__()
        self.item_embed = nn.Embedding(n_items, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, n_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, n_items)
    
    def forward(self, session_items):
        """
        session_items: (batch, seq_len) — item IDs in session order
        """
        x = self.item_embed(session_items)  # (batch, seq, embed_dim)
        out, hidden = self.gru(x)  # out: (batch, seq, hidden_dim)
        last_hidden = out[:, -1, :]  # (batch, hidden_dim)
        logits = self.output_layer(last_hidden)  # (batch, n_items)
        return logits
```

Training: next-item prediction with cross-entropy loss.

**What breaks:** GRU has a recency bias — items earlier in the session have attenuated influence on the hidden state. But sometimes the first item in a session is the most important signal.

### 13.3 Attention-Based Models: SASRec and BERT4Rec

**The problem:** GRU has a recency bias — items earlier in the session have attenuated influence on the hidden state. But sometimes the first item in a session is the most important signal.

**The core insight:** Self-attention can attend to any item in the session regardless of position. Long-range dependencies in session behavior are captured without the information bottleneck of a hidden state.

**SASRec (Kang & McAuley, 2018):**

```python
class SASRec(nn.Module):
    def __init__(self, n_items, embed_dim=128, n_heads=4, n_layers=2, max_len=50):
        super().__init__()
        self.item_embed = nn.Embedding(n_items, embed_dim, padding_idx=0)
        self.pos_embed = nn.Embedding(max_len, embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, 
            dim_feedforward=256, dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_layer = nn.Linear(embed_dim, n_items)
    
    def forward(self, session_items):
        seq_len = session_items.shape[1]
        positions = torch.arange(seq_len, device=session_items.device).unsqueeze(0)
        
        x = self.item_embed(session_items) + self.pos_embed(positions)
        
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len)
        
        out = self.transformer(x, mask=mask)
        last = out[:, -1, :]
        return self.output_layer(last)
```

**BERT4Rec:** Applies BERT-style masked language modeling to item sequences. Mask random items in the sequence and train the model to predict the masked items. More data-efficient than next-item prediction because each sequence generates multiple training examples.

### 13.4 Graph-Based Session Models

**SR-GNN (Wu et al., 2019):** Model each session as a directed graph, apply GNN to capture complex item transitions (including repeated visits and non-sequential paths), then predict the next item using an attention mechanism over the session graph representation.

**The problem SR-GNN solves:** Sessions sometimes contain non-sequential patterns: a user views A, then B, then returns to A, then views C. RNNs treat this as a linear sequence; a graph can represent the revisit explicitly.

**The core insight:** Model each session as a directed graph where nodes are items and edges are transitions. GNN aggregation over this graph captures complex intra-session item relationships.

### 13.5 Hybrid: Combining Session with Long-Term History

**The problem:** When does the session override long-term history, and when does long-term history override the session? A static blend misses context-dependent switching.

**The core insight:** Learn a gating function that determines how much to trust the session encoding vs. the long-term encoding based on session length, recency, and other context signals.

```
final_representation = α · session_encoding + (1 - α) · long_term_encoding

where α is learned from context (session length, recency, etc.)
```

**What breaks:** Session-based models are trained on sequential co-occurrence, which can be spurious (user accidentally clicked on two unrelated items in sequence). They also overfit to popular item sequences and fail to surface tail items. The model predicts "what usually comes next" — not always the same as "what this user needs next."

---

## 14. Common Interview Questions with Answers

### Q1: How would you design a recommendation system for Netflix from scratch?

**Answer:**

Requirements: ~250M users, ~15K titles, optimize for watch time and retention.

**Phase 1 — MVP:**
- Collect explicit feedback (ratings) and implicit feedback (watch history, completion rate, search queries)
- Item-based CF with cosine similarity on watch history
- Simple popularity-based fallback for cold start
- Daily batch scoring, serve from cache

**Phase 2 — Scale:**
- Matrix factorization (ALS in Spark) for user/item embeddings
- Separate retrieval (ANN over embeddings) from ranking (gradient boosted tree on dense features)
- Feature store for consistent training/serving features
- A/B testing infrastructure

**Phase 3 — Sophistication:**
- Two-tower neural model trained with in-batch negatives
- Content-based features from video metadata, audio/visual embeddings
- Context-aware: device, time of day, day of week, recent session
- Diversity/calibration in re-ranking
- Continuous training on fresh interaction data

**Metrics:** Offline: NDCG@10, Recall@100 (retrieval). Online: watch time, 30-day retention.

---

### Q2: Explain the difference between collaborative filtering and content-based filtering. When would you use each?

**Answer:**

**CF:** Uses interaction patterns. "Users who behaved like you liked X." No need to understand what items are — just how people interact with them. Requires interaction history; fails for cold items and cold users.

**CB:** Uses item features. "Items similar to what you liked." Works for cold items (just need features). Tends to over-specialize — cannot discover items outside known taste.

**Use CF when:** Dense interaction data, you want serendipity, item features are hard to engineer.

**Use CB when:** New item catalog, explainability required, sparse user data.

**In practice:** Use both. Hybrid systems dominate production.

---

### Q3: What is matrix factorization and why does it work?

**Answer:**

MF decomposes the user-item interaction matrix R into two lower-dimensional matrices: P (users × k) and Q (items × k), such that R ≈ PQ^T.

The k latent dimensions capture hidden factors that explain user preferences — things like "affinity for action movies," "preference for indie content." No one labels these; the model discovers them from observed interactions.

It works because the interaction matrix has low-rank structure: most user behavior can be explained by a small number of taste dimensions. 100M users might have complex individual histories, but their patterns of taste cluster into maybe 100-200 meaningful dimensions.

Optimization: minimize ||R - PQ^T||² over observed entries, with L2 regularization to prevent overfitting. Solved with SGD or ALS.

Key practical details: add bias terms (b_u, b_i), use k=64 to 512, and for implicit feedback use confidence-weighted loss (iALS): c_ui = 1 + α·f_ui.

---

### Q4: How do two-tower models work? What makes them suitable for large-scale retrieval?

**Answer:**

Two-tower models have separate neural networks for users and items. Both towers output a fixed-dimensional embedding. The compatibility score is computed as a dot product between the user and item embeddings.

The critical property: the user and item towers never interact except at the final dot product. This means all item embeddings can be precomputed offline, at query time you compute only the user embedding (one forward pass), and use ANN (FAISS, ScaNN) to find the top-K items.

This turns a problem that would require 100M forward passes into one forward pass + one ANN lookup.

Training: in-batch negatives with cross-entropy loss (sampled softmax). Challenge: popularity bias from in-batch negatives. Fix: subtract log(p_i) from item scores.

---

### Q5: Explain the cold start problem and how you would address it.

**Answer:**

Cold start occurs when a user or item has no interaction history for collaborative filtering to work.

**New user:**
- Onboarding survey (pick 3-5 preferences)
- Demographic-based proxies
- Popular items fallback
- Exploration slate designed to quickly learn taste
- Cross-platform transfer

**New item:**
- Content-based features → embed in item space immediately
- Similar item lookup: find nearest established items by features
- Warm-up injection: show new items to a random fraction of users
- Feature-based two-tower: item tower takes features (not ID), so new items get an embedding from day 1

The deeper fix: use item feature representations (not just IDs) in your model so new items are never completely cold.

---

### Q6: How would you evaluate a recommender system?

**Answer:**

**Offline metrics:** NDCG@K (ranking quality, position-discounted), Precision@K and Recall@K (set quality), MAP (average precision across users), MRR (rank of first hit), HR@K (binary hit metric).

**Online (A/B test):** CTR, watch time, retention. The ground truth for launch decisions.

**Why both:** Offline is cheap and fast for development. Online is expensive and slow but measures what actually matters.

**Common trap:** Offline gains do not always translate to online gains. Always A/B test before launch. Watch for novelty effects inflating short-term A/B engagement.

For retrieval specifically, use Recall@K — if the right items do not make it into the candidate set, downstream ranking cannot help.

---

### Q7: How do you handle popularity bias in recommendations?

**Answer:**

Popularity bias: popular items get recommended more → more interactions → ranked even higher → feedback loop. Long-tail items never get exposure.

**Causes:** Training data has more signal for popular items. In-batch negatives over-represent popular items. Ranking features like "total plays" discriminate against new items.

**Fixes:**
- **Inverse propensity scoring (IPS):** Weight each training example by 1/p(item seen). Debiases the training objective.
- **Sampling correction in two-tower:** Subtract log(popularity) from item scores.
- **Exploration slots:** Reserve a fraction of recommended positions for long-tail items.
- **Diversity constraints:** Cap the fraction of popular items per slate.
- **Separate popularity features from personalization features** in the ranking model.

---

### Q8: What is LambdaRank and why is it better than pointwise ranking?

**Answer:**

Pointwise ranking treats each (user, item) pair independently and minimizes MSE or BCE on individual relevance scores. It does not care about relative order.

LambdaRank directly optimizes the gradient of NDCG, even though NDCG is not differentiable.

Key insight: define what the gradient should be without defining a differentiable loss. For a pair of items (i, j) where i should rank above j:

```
λ_ij = ∂L/∂(s_i - s_j) = -σ/(1 + e^{σ(si-sj)}) · |ΔNDCG_{ij}|
```

The |ΔNDCG| term scales the gradient by how important the swap is. Swapping items at ranks 1 and 2 gets a much larger gradient than swapping ranks 50 and 51.

**LambdaMART** = LambdaRank + gradient boosted trees. Widely used in production.

**Why better:** Directly optimizes ranking quality at the top, not a surrogate that loosely correlates with it.

---

### Q9: How do you design for diversity in recommendations?

**Answer:**

Pure relevance optimization leads to a homogeneous list. Users report higher satisfaction with a diverse list.

1. **MMR:** Greedily select items that maximize λ·relevance - (1-λ)·max_similarity_to_selected.
2. **DPP (Determinantal Point Processes):** Sample a set of K items rewarding both quality and diversity. Principled but computationally expensive.
3. **Calibration:** KL divergence penalty between recommendation distribution and user's historical consumption distribution across categories.
4. **Hard constraints:** No more than 3 items of the same genre in a 10-item slate.
5. **Re-ranking layer:** Let ranking focus on quality; enforce diversity as post-processing.

The λ parameter is tuned via A/B testing, not offline metrics.

---

### Q10: Walk me through how YouTube's recommendation system works.

**Answer:**

Based on Covington et al. (2016):

**Stage 1 — Candidate Generation (Two-Tower):**
- User tower: average embedding of watch history + search tokens + demographics + context (time of day, device)
- Item tower: video features, title, tags
- Trained to predict next video watch
- ANN retrieval fetches ~200 candidates from millions

**Stage 2 — Ranking:**
- Richer features: user-video pair features not feasible at retrieval scale
- Video age feature: prevents model from learning "old videos are better because we have observed them long enough"
- Impressions feature: have we shown this to this user before?
- Multi-objective: click, watch time, likes
- Weighted by watch time, not clicks — avoids clickbait

**Key design choices:**
- Asymmetric co-watch: treat the most recent watch as label, prior watches as context → models "what next"
- "Example age" feature: model explicitly learns freshness

---

### Q11: How would you handle a recommender system that has been deployed but performance is degrading?

**Answer:**

**Diagnose first:**
1. Which metric is degrading? CTR? NDCG? Coverage? A specific user segment?
2. Check for data drift: has the input feature distribution shifted? (PSI monitoring)
3. Check for label/feedback shift: has the click-through ratio on organic traffic changed?
4. Check the data pipeline: stale features in the feature store? Missing features returning defaults?

**Common causes:**
- **Concept drift:** User taste and trends change. Model trained in January does not know about trends in August.
- **Feedback loop:** Recommendations influence what users see → what users interact with → what the model trains on → distribution collapse.
- **Data quality:** Missing features, pipeline bugs, upstream schema changes.

**Remediation:**
- Retrain on fresh data (daily or weekly for fast-moving domains)
- Monitor feature importance: if important features went missing, fix the pipeline
- Add freshness features explicitly (item age, trending score)
- Evaluate training data distribution across user segments

---

### Q12: BPR loss — write it out and explain the intuition.

**Answer:**

For each user u, positive item i, negative item j:

```
L_BPR = -Σ_{(u,i,j)} log σ(r̂_ui - r̂_uj) + λ||Θ||²
```

**Intuition:** We want the score for the positive item to be higher than the score for the negative item. σ(r̂_ui - r̂_uj) is the probability of ranking i above j. Maximize this probability → minimize its negative log.

**Why better than pointwise BCE:** We do not assume we know the "true" relevance of an item, just that i is more relevant than j for user u. This is a weaker and more reasonable assumption from implicit feedback. The model is explicitly trained to rank correctly, not to predict absolute scores.

```python
def bpr_loss(model, user_ids, pos_item_ids, neg_item_ids, reg=0.01):
    pos_scores = model(user_ids, pos_item_ids)
    neg_scores = model(user_ids, neg_item_ids)
    
    loss = -torch.log(torch.sigmoid(pos_scores - neg_scores)).mean()
    
    reg_loss = reg * (
        model.user_embed.weight[user_ids].norm(2).pow(2) +
        model.item_embed.weight[pos_item_ids].norm(2).pow(2) +
        model.item_embed.weight[neg_item_ids].norm(2).pow(2)
    ).mean()
    
    return loss + reg_loss
```

---

## Quick Reference: Formula Sheet

```
# Matrix Factorization
r̂(u,i) = μ + b_u + b_i + p_u^T q_i

# BPR Loss
L = -Σ log σ(r̂_ui - r̂_uj) + λ||Θ||²

# Two-Tower Loss (in-batch negatives)
L = CrossEntropy(logits / τ, diagonal_labels)
logits_ij = user_i^T · item_j

# DCG@K
DCG@K = Σ_{i=1}^K (2^{rel_i} - 1) / log_2(i+1)

# NDCG@K
NDCG@K = DCG@K / IDCG@K

# Precision@K
P@K = |{relevant} ∩ {top-K}| / K

# Recall@K
R@K = |{relevant} ∩ {top-K}| / |{relevant}|

# MAP
MAP = mean_u [ (1/|rel_u|) · Σ_k P@k · rel(k) ]

# LightGCN propagation
e_u^(k) = Σ_{i ∈ N(u)} (1/√|N(u)||N(i)|) · e_i^(k-1)

# ALS update
p_u = (Q^T Q + λI)^{-1} Q^T r_u

# MMR
MMR = arg max_{i ∉ S} [λ·sim(i, q) - (1-λ)·max_{j ∈ S} sim(i,j)]

# Confidence in iALS
c_ui = 1 + α·f_ui
```

---

## Key Papers to Know

| Paper | Contribution |
|-------|-------------|
| Koren et al. (2009) — Matrix Factorization Techniques | SVD/MF for RS, Netflix Prize approach |
| Hu, Koren, Volinsky (2008) — iALS | Implicit feedback MF with confidence weighting |
| Rendle et al. (2009) — BPR | Pairwise ranking loss for implicit feedback |
| He et al. (2017) — NCF | Neural collaborative filtering, MLP over embeddings |
| Covington et al. (2016) — YouTube DNN | Two-stage retrieval+ranking at scale |
| He et al. (2020) — LightGCN | Simplified GCN for RS, SOTA on many benchmarks |
| Kang & McAuley (2018) — SASRec | Self-attention for sequential recommendation |
| Wu et al. (2019) — SR-GNN | Graph NN for session-based recommendation |
| Sun et al. (2019) — BERT4Rec | BERT for sequential item prediction |
| Linden et al. (2003) — Amazon item-to-item CF | Item-based CF at Amazon scale |
