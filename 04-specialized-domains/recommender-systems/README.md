# Recommender Systems: A Deep Dive for ML Interviews

> "A recommender system is that knowledgeable friend who has watched everything on Netflix, listened to every song on Spotify, and bought everything on Amazon — and actually remembers what you liked and why."

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

### The Knowledgeable Friend Analogy

Imagine you move to a new city and you know someone who has eaten at every restaurant, seen every movie at every theater, and read every book at every bookstore. When you say "I'm in the mood for something spicy but not too heavy," they don't hand you a Yelp printout sorted by rating — they say "You'd love that little Thai place on 5th, it's nothing like the heavy curries you didn't like last time."

That is what a recommender system aspires to be. Not a search engine (you already know what you want), not a popularity chart (everyone else's opinion), but a personalized oracle that understands *your* taste and maps it onto an enormous catalog.

### The Scale Problem

Without recommendations:
- Netflix has 15,000+ titles. The average user watches 2 hours/day. Without guidance, discovery fails.
- Spotify has 100M+ tracks. The average listener hears maybe 30 songs/day.
- Amazon has 350M+ products. A browsing session might cover 20 items.

The job of a recommender system is to collapse that impossibly large catalog into a short list that feels personally curated. Netflix has famously said that 80% of content watched is discovered through recommendations rather than search.

### Business Value

| Platform | Estimated recommendation value |
|----------|--------------------------------|
| Netflix | $1B/year in retained subscribers |
| Amazon | 35% of revenue from recommendations |
| Spotify | Core to user retention and discovery |
| YouTube | 70% of watch time from recommended content |

### The Core Formulation

Given a set of users U, a set of items I, and a (sparse) matrix of observed interactions R, predict the "score" r̂(u, i) for all unobserved (u, i) pairs and surface the top-K highest-scoring items per user.

```
R : |U| × |I|  matrix   (very sparse — typically 99%+ empty)
Goal: fill in the blanks in a way that maximizes user satisfaction
```

---

## 2. Types of Recommender Systems

Think of it this way: to recommend music, you can either:
- Ask people with similar tastes what they liked (*collaborative*)
- Analyze the music itself — tempo, key, genre (*content-based*)
- Do both (*hybrid*)

### 2.1 Collaborative Filtering (CF)

**The wisdom of crowds approach.** No need to understand what an item *is* — just find people whose past behavior matches yours, and recommend what they liked.

- "Users who bought this also bought..."
- "Because you watched Stranger Things, you might like Dark"

Requires: interaction data. No item features needed.

### 2.2 Content-Based Filtering (CB)

**The item DNA approach.** Build a profile of what items look like (genres, tags, text, audio features) and match them to a user's preference profile.

- Spotify's early recommendation was heavily content-based: analyze the audio waveform
- News recommenders: TF-IDF on article text

Requires: item features. Less sensitive to cold start for new items.

### 2.3 Hybrid Systems

Most production systems are hybrid. Netflix uses:
1. CF to find neighbors
2. Content features to break ties and handle cold start
3. Contextual signals (time of day, device, recent session)
4. Ranking models on top

```
Score(u, i) = α · CF_score(u, i) + β · CB_score(u, i) + γ · Context_score(u, i)
```

The weights α, β, γ can themselves be learned.

### 2.4 Knowledge-Based Systems

Rule-driven, often for complex or infrequent purchases (mortgages, cars). "Given your constraints (budget, family size, fuel preference), here are your options." Less ML, more constraint satisfaction. Not covered in depth here.

---

## 3. Collaborative Filtering Deep Dive

### 3.1 User-Based CF

**The idea:** Find users who behaved like you in the past. Recommend what they liked that you haven't seen yet.

**The Amazon circa-2003 version:**
1. Compute similarity between your purchase/rating history and every other user
2. Pick the top-K most similar users (your "neighborhood")
3. Aggregate their ratings/interactions for items you haven't touched
4. Recommend the highest-scoring items

**Similarity Metrics:**

Cosine similarity (treats ratings as vectors):
```
sim(u, v) = (r_u · r_v) / (||r_u|| · ||r_v||)
```

Pearson correlation (accounts for rating bias — some users always rate high):
```
sim(u, v) = Σ_i (r_ui - r̄_u)(r_vi - r̄_v) / √[Σ_i(r_ui - r̄_u)² · Σ_i(r_vi - r̄_v)²]
```

**Prediction:**
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

**Scaling problem:** With 100M users, computing pairwise similarities is O(|U|²) — impossible in real time. This led to item-based CF and eventually matrix factorization.

### 3.2 Item-Based CF

**The Amazon breakthrough (Linden et al., 2003).** Instead of finding similar users, find similar items. Item-item similarity is more stable (items don't change behavior), and there are usually fewer items than users.

Key insight: item-item similarities can be precomputed offline. At query time, look up the items a user interacted with, find their neighbors, aggregate.

```
score(u, i) = Σ_{j ∈ rated_by_u} sim(i, j) · r(u, j) / Σ_{j} |sim(i, j)|
```

**Why item-based often beats user-based in practice:**
- Items have more interactions than individual users → better similarity estimates
- More stable: "The Dark Knight is similar to Inception" doesn't change
- O(|I|²) precomputation, O(K) online lookup

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
    # Zero out already-interacted items
    for item_id in user_history:
        scores[item_id] = 0
    return np.argsort(scores)[::-1][:top_k]
```

### 3.3 Matrix Factorization

**The paradigm shift.** Instead of explicit neighborhoods, learn latent representations. Decompose the rating matrix into user and item embedding matrices.

**The intuition:** Think of latent factors as hidden "taste dimensions." For movies, these might loosely correspond to:
- How action-packed is it?
- How artsy vs. mainstream?
- How romantic?
- How long is it?

No one labels these dimensions — the model discovers them from data.

```
R ≈ P · Q^T

P: (|U|, k)  — user latent factors
Q: (|I|, k)  — item latent factors
k: number of latent dimensions (typically 64–512)

r̂(u, i) = p_u · q_i  (dot product)
```

**SVD (Singular Value Decomposition):**

The "textbook" approach. Decompose R = UΣV^T, truncate to k dimensions.

```
Problem: R is sparse — SVD assumes you know all values.
Fix: only factorize observed entries, with regularization.
```

**Regularized MF (the actual workhorse):**

```
min_{P, Q} Σ_{(u,i) observed} (r_ui - p_u^T q_i)² + λ(||p_u||² + ||q_i||²)
```

Solved by stochastic gradient descent (SGD):

```python
# SGD update for a single (user, item, rating) triple
def sgd_update(p_u, q_i, r_ui, lr=0.01, reg=0.02):
    error = r_ui - np.dot(p_u, q_i)
    
    p_u_new = p_u + lr * (error * q_i - reg * p_u)
    q_i_new = q_i + lr * (error * p_u - reg * q_i)
    
    return p_u_new, q_i_new

# Training loop
for epoch in range(n_epochs):
    for u, i, r in observed_ratings:
        p[u], q[i] = sgd_update(p[u], q[i], r)
```

**Bias terms (important in practice):**

Users and items have global biases. Some users rate everything high; some movies are universally loved/hated.

```
r̂(u, i) = μ + b_u + b_i + p_u^T q_i
```

Where μ is the global mean rating, b_u is user bias, b_i is item bias.

### 3.4 ALS (Alternating Least Squares)

**Why ALS instead of SGD?** ALS alternates between fixing Q and solving for P exactly (and vice versa). Each sub-problem is a convex least-squares problem with a closed-form solution.

```
Fix Q → solve for each p_u:
    p_u = (Q^T Q + λI)^{-1} Q^T r_u

Fix P → solve for each q_i:
    q_i = (P^T P + λI)^{-1} P^T r_i
```

**ALS advantages:**
- Parallelizes beautifully: each user/item update is independent
- No learning rate to tune
- Works well for implicit feedback (see below)
- Spark MLlib's ALS is the go-to for large-scale MF

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

### 3.5 Implicit vs. Explicit Feedback

**Explicit feedback:** Star ratings, thumbs up/down. Clear signal, but:
- Very sparse (most users never rate anything)
- Selection bias (you only rate things you watched; you watched things you expected to like)
- Netflix famously found that what users *watch* predicts future behavior better than what they *rate*

**Implicit feedback:** Clicks, views, purchases, streams, time spent. Much denser, but:
- Noisy (you might click on something you hate)
- No negatives (you didn't click on X — but why? Didn't see it? Didn't want it? Already own it?)
- Confidence matters: streaming a song 50 times is stronger signal than streaming it once

**Hu, Koren, Volinsky (2008) — iALS:**

Model confidence in an implicit observation:
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

This treats *all* (user, item) pairs as training examples — zero interactions get weight c=1, positive interactions get higher confidence.

---

## 4. Content-Based Filtering

### 4.1 The Core Idea

Build a feature vector for each item. Build a preference profile for each user (derived from items they liked). Score new items by matching them against the user profile.

Spotify early approach: "You like songs with fast tempo, minor key, high energy, electronic instrumentation → here are more songs with those properties."

### 4.2 TF-IDF for Text-Based Items

Great for news articles, product descriptions, movie plots.

**Term Frequency (TF):** How often does word w appear in document d?
```
TF(w, d) = count(w in d) / total_words(d)
```

**Inverse Document Frequency (IDF):** How rare is word w across all documents?
```
IDF(w) = log(N / df(w))

N = total documents
df(w) = documents containing w
```

**TF-IDF:**
```
TF-IDF(w, d) = TF(w, d) · IDF(w)
```

Words like "the" get near-zero IDF. Genre-specific words like "heist" or "spaceship" get high IDF — they're discriminative.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Build item feature matrix
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

# Item similarity
item_sim = cosine_similarity(item_matrix)

# User profile: average TF-IDF of liked items
def build_user_profile(liked_item_indices, item_matrix):
    liked_vecs = item_matrix[liked_item_indices]
    return liked_vecs.mean(axis=0)

# Score items for user
def score_items(user_profile, item_matrix, already_seen):
    scores = cosine_similarity(user_profile, item_matrix).flatten()
    scores[already_seen] = 0
    return scores.argsort()[::-1][:10]
```

### 4.3 Feature Engineering for Different Domains

**Movies/TV:**
- Genre (one-hot)
- Cast/director (entity embeddings)
- Plot synopsis (TF-IDF or sentence embeddings)
- Release year, runtime
- MPAA rating

**Music (Spotify):**
- Audio features: tempo, energy, danceability, valence, loudness
- Genre tags
- Artist embeddings
- Lyrics (NLP features)

**E-commerce (Amazon):**
- Category hierarchy
- Brand, price tier
- Product attributes (size, color, material)
- Image embeddings (visual similarity)

**The representation bottleneck:** All these heterogeneous features need to map into a unified embedding space for comparison. This is why neural methods (Section 5, 6) took over — they learn the feature mapping jointly with the recommendation task.

### 4.4 Content-Based Pros and Cons

| Pros | Cons |
|------|------|
| No cold start for new items | Cold start for new users |
| Explainable ("because you liked action movies") | Feature engineering burden |
| No popularity bias | Can't discover items outside user's known taste |
| Works for niche tastes | Overspecialization — no serendipity |

---

## 5. Neural Collaborative Filtering (NCF)

### 5.1 Motivation

Standard MF uses a dot product to combine user and item embeddings:
```
r̂(u, i) = p_u^T q_i = Σ_k p_uk · q_ik
```

This is linear. The dot product can't capture complex, non-linear interaction patterns. What if "user likes sci-fi" and "item is hard sci-fi" interact in a more nuanced way than a simple product?

**NCF (He et al., 2017)** replaces the dot product with a neural network.

### 5.2 Architecture

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
        
        # Final prediction layer
        self.predict = nn.Linear(embed_dim + mlp_layers[-1], 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, user_ids, item_ids):
        # GMF branch
        gmf_user = self.gmf_user_embed(user_ids)
        gmf_item = self.gmf_item_embed(item_ids)
        gmf_out = gmf_user * gmf_item  # element-wise product
        
        # MLP branch
        mlp_user = self.mlp_user_embed(user_ids)
        mlp_item = self.mlp_item_embed(item_ids)
        mlp_input = torch.cat([mlp_user, mlp_item], dim=-1)
        mlp_out = self.mlp(mlp_input)
        
        # NeuMF: combine both branches
        combined = torch.cat([gmf_out, mlp_out], dim=-1)
        output = self.sigmoid(self.predict(combined))
        return output.squeeze()
```

### 5.3 Training with Negative Sampling

Implicit feedback means we only observe positives. We need negative examples.

```python
def sample_negatives(pos_user_item_pairs, n_items, n_neg=4):
    """For each positive (u, i) pair, sample n_neg random items not in user's history."""
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

# Binary cross-entropy loss
criterion = nn.BCELoss()
```

### 5.4 NCF vs. MF: When to Use Which?

- MF: simpler, faster, often competitive, easier to interpret
- NCF: can capture non-linear patterns, better for complex interactions
- In practice, the gap is smaller than you'd expect — good features and regularization matter more than model complexity

---

## 6. Two-Tower Models

### 6.1 The Scale Problem

NCF (with concatenation + MLP) requires a forward pass for every (user, item) pair at serving time. With 100M items, that's 100M forward passes per user query — completely infeasible.

Two-tower models solve this with a key architectural constraint: **the user and item towers are separate networks that only interact at the final dot product.**

This enables pre-computation of all item embeddings offline, and fast approximate nearest neighbor (ANN) search at query time.

### 6.2 Architecture

```
User features ─→ User Tower (MLP) ─→ user_embedding (d-dim)
                                              ↘
                                               dot product ─→ score
                                              ↗
Item features ─→ Item Tower (MLP) ─→ item_embedding (d-dim)
```

YouTube, Google, Pinterest, and most large-scale recommenders use this architecture.

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
            nn.LayerNorm(embed_dim)  # L2 normalize for cosine similarity
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
        
        # Cosine similarity (embeddings are L2-normalized)
        scores = torch.sum(user_emb * item_emb, dim=-1)
        return scores
    
    def get_user_embedding(self, user_features):
        return self.user_tower(user_features)
    
    def get_item_embedding(self, item_features):
        return self.item_tower(item_features)
```

### 6.3 Training

**In-batch negatives** are key. For a batch of B (user, positive_item) pairs, treat every other item in the batch as a negative for each user. This gives B-1 negatives "for free."

```python
def two_tower_loss(user_embs, item_embs, temperature=0.07):
    """
    user_embs: (B, d)
    item_embs: (B, d) — one positive item per user
    """
    # Compute all pairwise scores: (B, B)
    logits = torch.matmul(user_embs, item_embs.T) / temperature
    
    # Labels: diagonal is positive (user i matched with item i)
    labels = torch.arange(logits.shape[0]).to(logits.device)
    
    # Cross-entropy loss
    loss = nn.CrossEntropyLoss()(logits, labels)
    return loss
```

**Sampling bias correction:** Popular items appear more often in batches as negatives, causing the model to down-weight popular items more than it should. Fix: subtract log(p_i) from item scores where p_i is the item's sampling probability.

```
corrected_score(u, i) = score(u, i) - log(p_i)
```

### 6.4 Retrieval with ANN

Once trained, precompute all item embeddings and build an ANN index.

```python
import faiss
import numpy as np

# Build index (offline, once)
d = 128  # embedding dimension
item_embeddings = get_all_item_embeddings()  # (n_items, d)

index = faiss.IndexFlatIP(d)  # Inner product (cosine if normalized)
# For large scale, use IVF or HNSW:
# index = faiss.IndexIVFFlat(faiss.IndexFlatIP(d), d, n_clusters)
index.add(item_embeddings.astype(np.float32))

# Query (online, per user request)
def retrieve_candidates(user_features, k=500):
    user_emb = model.get_user_embedding(user_features)
    user_emb = user_emb.detach().numpy().astype(np.float32)
    
    distances, indices = index.search(user_emb.reshape(1, -1), k)
    return indices[0], distances[0]
```

### 6.5 Two-Stage Architecture

The two-tower retrieves candidates (top-500 to top-1000). A separate, more expensive **ranking model** then scores those candidates with richer features.

```
All items (100M)
     ↓ [Two-Tower ANN retrieval]
Candidate set (~500 items)
     ↓ [Ranking model (DNN, LightGBM, etc.)]
Ranked list (top 20)
     ↓ [Re-ranking for diversity, business rules]
Final slate shown to user
```

This is the standard pipeline at YouTube, Netflix, Spotify, LinkedIn, and virtually every large-scale recommender.

### 6.6 YouTube Deep Neural Net (Covington et al., 2016)

The canonical two-tower paper. Key design choices:
- **Candidate generation:** User tower takes watch history (average of video embeddings), search tokens, demographics, context. Item tower: video features.
- **Serving:** Nearest neighbor lookup, not dot product at query time
- **Example age feature:** Time since video was uploaded — prevents the model from learning "old videos are good because we've observed them long enough"
- **Asymmetric co-watch:** Training treats the most recently watched video as the label, with prior watches as context — predicts "what will you watch *next*"

---

## 7. Learning to Rank

### 7.1 Why Ranking Matters

Retrieval gives you candidates. Ranking decides the order. The first result gets 10x more clicks than the 5th. Getting the ranking right is often more valuable than improving retrieval.

### 7.2 Three Paradigms

#### 7.2.1 Pointwise

Treat ranking as regression or binary classification. Predict the relevance score of each (query, item) pair independently. The loss function doesn't care about relative order.

```
Loss = Σ_{u,i} (r̂(u,i) - r(u,i))²  [MSE for ratings]
     or
Loss = Σ_{u,i} -[r(u,i)·log(r̂(u,i)) + (1-r(u,i))·log(1-r̂(u,i))]  [BCE for implicit]
```

**Problem:** Predicting "absolute" relevance scores independently doesn't optimize what we care about — the *order*. You can predict all scores perfectly wrong if you're systematically off.

#### 7.2.2 Pairwise

For each user, create pairs of items where one is more relevant than the other. Minimize the probability of ranking the less relevant item higher.

```python
# RankNet (Burges et al., 2005)
def ranknet_loss(score_i, score_j, label):
    """
    score_i, score_j: predicted scores for items i and j
    label: 1 if i should rank above j, 0 otherwise
    """
    diff = score_i - score_j
    # Probability that i ranks above j
    p_ij = torch.sigmoid(diff)
    return nn.BCELoss()(p_ij, label.float())
```

**BPR (Bayesian Personalized Ranking):** Widely used in RS. For each user u, sampled positive item i and negative item j:

```
L_BPR = -Σ_{(u,i,j)} log σ(r̂_ui - r̂_uj) + λ||Θ||²
```

```python
def bpr_loss(pos_scores, neg_scores, reg_lambda=0.01):
    """
    pos_scores: (B,) scores for positive items
    neg_scores: (B,) scores for negative items
    """
    diff = pos_scores - neg_scores
    loss = -torch.log(torch.sigmoid(diff)).mean()
    return loss
```

#### 7.2.3 Listwise

Optimize a loss function defined over the entire ranked list. Directly optimizes ranking metrics.

**SoftMax / ListNet:**
```
Loss = -Σ_i P_true(i) · log P_model(i)

P_model(i) = exp(score_i) / Σ_j exp(score_j)
```

**LambdaRank / LambdaMART:**

The key insight: you don't need a loss function — you need *gradients*. LambdaRank defines the gradient directly from the change in NDCG if items i and j were swapped:

```
λ_ij = ∂L/∂(s_i - s_j) = -σ / (1 + exp(σ(s_i - s_j))) · |ΔNDCG|
```

The |ΔNDCG| term weights the gradient by "how much would this swap improve the ranking?" Swapping items at rank 1 and 2 matters more than rank 99 and 100.

**LambdaMART** = LambdaRank + MART (Multiple Additive Regression Trees = gradient boosted trees). The standard workhorse at Microsoft/Bing and many production ranking systems.

```python
# LightGBM LambdaMART (common production setup)
import lightgbm as lgb

params = {
    'objective': 'lambdarank',
    'metric': 'ndcg',
    'ndcg_eval_at': [1, 3, 5, 10],
    'num_leaves': 31,
    'min_data_in_leaf': 20,
    'num_iterations': 500,
    'learning_rate': 0.05,
    'label_gain': [0, 1, 3, 7, 15]  # gains for relevance levels 0-4
}

train_data = lgb.Dataset(X_train, label=y_train, group=group_train)
model = lgb.train(params, train_data, valid_sets=[valid_data])
```

### 7.3 Summary Comparison

| Approach | Optimizes | Typical Use | Pros | Cons |
|----------|-----------|-------------|------|------|
| Pointwise | Individual relevance | Rating prediction | Simple | Ignores order |
| Pairwise (BPR) | Relative order | Top-K ranking | Better than pointwise | Doesn't optimize full list quality |
| Listwise (LambdaMART) | List quality (NDCG) | Production ranking | Best metrics | Slower, harder to implement |

---

## 8. The Cold Start Problem

### 8.1 The Two Flavors

**New User Cold Start:** Netflix doesn't know anything about you on day 1. What do you recommend?

**New Item Cold Start:** A new Netflix original premieres. No one has watched it yet. How do you surface it?

This is where pure collaborative filtering fails completely — you need interactions to make predictions, but items/users without interactions can't participate.

### 8.2 New User Strategies

**Onboarding questionnaires:** Spotify's "pick 3 artists you like." Explicit preference elicitation. Fast, works, but users hate long surveys.

**Demographic-based:** Use age, location, device type to find similar users who've been around longer. Crude but better than nothing.

**Popular items fallback:** Recommend the most popular items in broad categories. The "safe" default — you won't be wildly wrong, but not personalized.

**Cross-domain transfer:** If users have a social media profile linked, or if they import ratings from another platform, bootstrap from that.

**Exploration / exploit:** Use first few interactions aggressively. Show a diverse set on day 1 specifically designed to maximize information gain about user taste, not maximize immediate click rate.

```python
def cold_start_explore_slate(user, session_history, all_items, n=20):
    """
    For a cold-start user, recommend a diverse slate that maximizes 
    information gain about their taste.
    """
    # Cover the major taste clusters
    cluster_centers = get_cluster_centers(all_items)
    
    # Pick one high-quality item from each cluster
    candidates = []
    for cluster_id in range(n):
        top_item = get_top_item_in_cluster(cluster_id, min_popularity=100)
        candidates.append(top_item)
    
    return candidates[:n]
```

### 8.3 New Item Strategies

**Content-based features:** New item? Extract its features (genre, description, audio features) and find similar established items. Use those similar items' user base to seed recommendations.

```
new_item_embedding ≈ average(embeddings of k most similar existing items)
```

**Warm-up phase:** Explicitly inject new items into recommendation slates at a small rate (e.g., 1 slot in 20 is reserved for new item exploration). Collect interaction data. Graduate to full CF once enough data is accumulated.

**Context-aware injection:** Show new items in contexts where they're most likely to succeed. New action movie? Inject it for users currently in an "action movie mood" (based on recent session).

**Side information for embeddings:** Train item embeddings that can be computed from features alone (not just ID-based). New item's features → immediate embedding → ANN retrieval works on day 1.

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
    
    # At inference time for a new item:
    # embedding = encoder(extract_features(new_item))
    # Add to ANN index — done.
```

### 8.4 Meta-Learning Approaches

**MAML / FOMAML for cold start:** Frame as few-shot learning. Train the model to quickly adapt to a new user/item from a small number of examples. The model learns to learn.

This is an active research area. Practical in some production systems at large companies.

---

## 9. Diversity, Serendipity, and Filter Bubbles

### 9.1 The Echo Chamber Problem

Pure accuracy optimization creates a feedback loop:
1. User watches sci-fi → system recommends more sci-fi
2. User watches those → even more sci-fi recommended
3. User's profile collapses to a narrow taste cluster
4. User stops discovering new genres
5. User disengages → churn

Netflix found that pure watch-probability optimization led to users watching fewer unique titles over time. Optimizing only for immediate engagement hurts long-term retention.

### 9.2 Diversity

**Intra-list diversity:** The slate of K items should cover different taste dimensions, not 10 variants of the same genre.

**Coverage:** Across all users, what fraction of the catalog gets recommended? Popularity bias causes the tail to be invisible.

**Temporal diversity:** Don't show the same items every session.

**Measuring diversity:**

Intra-list diversity (ILD):
```
ILD(L) = (2 / (|L|(|L|-1))) · Σ_{i,j ∈ L, i≠j} (1 - sim(i, j))
```

**MMR (Maximal Marginal Relevance):** Greedily select items that balance relevance and diversity.

```python
def mmr_reranking(candidates, user_embedding, item_embeddings, 
                   already_selected=[], lambda_=0.5, k=10):
    """
    MMR: balance relevance to user vs. diversity from already selected items.
    """
    selected = list(already_selected)
    remaining = list(candidates)
    
    for _ in range(k):
        best_score = -float('inf')
        best_item = None
        
        for item in remaining:
            # Relevance to user
            relevance = cosine_similarity(
                user_embedding, item_embeddings[item]
            )
            
            # Similarity to already-selected items
            if selected:
                redundancy = max(
                    cosine_similarity(item_embeddings[item], item_embeddings[s])
                    for s in selected
                )
            else:
                redundancy = 0
            
            # MMR score
            mmr_score = lambda_ * relevance - (1 - lambda_) * redundancy
            
            if mmr_score > best_score:
                best_score = mmr_score
                best_item = item
        
        selected.append(best_item)
        remaining.remove(best_item)
    
    return selected
```

### 9.3 Serendipity

Serendipity = unexpected but pleasant. The difference between:
- **Expected:** You like action movies → here's another action movie (not serendipitous)
- **Serendipitous:** You like action movies, but you'd also love this quiet indie drama if you gave it a chance

Measuring serendipity is hard — it's fundamentally about user surprise combined with user satisfaction. Proxy: items that are dissimilar to the user's current profile but end up getting high ratings.

### 9.4 Filter Bubbles

The societal concern: recommendation algorithms may reinforce existing beliefs, preferences, and viewpoints by only showing users content that confirms what they already think.

Mitigation strategies:
- **Exploration budget:** Reserve a fixed fraction of slots for out-of-distribution recommendations
- **Diversity constraints:** Hard constraints on genre/topic distribution
- **Temporal freshness:** Ensure recently published content gets surfaced
- **Serendipity objectives:** Add serendipity as a term in the optimization objective

### 9.5 Calibration

User's recommended content should reflect their *breadth* of interests, not just their deepest interest.

If a user watches 60% action movies and 40% documentaries but only recommends action movies, calibration is broken.

```
Calibration loss = KL divergence(target distribution, recommendation distribution)
```

Where target distribution comes from the user's historical consumption across categories.

---

## 10. Evaluation Metrics

### 10.1 The Problem with MSE/MAE

Rating prediction (minimizing MSE on a held-out test set) does not measure what we care about: does the user engage with what we recommend? A system that predicts 4.1 stars vs. 4.2 stars for two top items — wrong order — but has low MSE is not actually good.

We care about **ranking quality**, specifically quality at the top of the list.

### 10.2 Precision@K

Of the K items recommended, what fraction are relevant?

```
Precision@K = |{relevant items} ∩ {top K recommended}| / K
```

```python
def precision_at_k(recommended, relevant, k):
    recommended_k = recommended[:k]
    hits = len(set(recommended_k) & set(relevant))
    return hits / k
```

**Problem:** Doesn't penalize a system for missing relevant items. If you have 100 relevant items and return 10, Precision@10 = 1.0 — but you missed 90.

### 10.3 Recall@K

Of all relevant items, what fraction appear in the top K?

```
Recall@K = |{relevant items} ∩ {top K recommended}| / |{relevant items}|
```

```python
def recall_at_k(recommended, relevant, k):
    recommended_k = recommended[:k]
    hits = len(set(recommended_k) & set(relevant))
    return hits / max(len(relevant), 1)
```

**Problem:** Doesn't account for rank order within the top K.

### 10.4 NDCG (Normalized Discounted Cumulative Gain)

The gold standard for ranking quality. Accounts for both relevance and position.

**DCG@K:**
```
DCG@K = Σ_{i=1}^{K} (2^{rel_i} - 1) / log_2(i + 1)
```

Relevance at position 1 is weighted at 1/log(2) = 1.0, position 2 at 1/log(3) ≈ 0.63, position 3 at 1/log(4) = 0.5, etc.

**IDCG@K:** The DCG of the ideal (perfect) ranking.

**NDCG@K = DCG@K / IDCG@K** — normalized to [0, 1].

```python
import numpy as np

def dcg_at_k(relevances, k):
    """
    relevances: list of relevance scores (e.g., [1, 0, 1, 0, 1]) for returned items
    """
    relevances = np.array(relevances[:k])
    if len(relevances) == 0:
        return 0
    discounts = np.log2(np.arange(2, len(relevances) + 2))
    return np.sum((2**relevances - 1) / discounts)

def ndcg_at_k(recommended, relevant_with_scores, k):
    """
    recommended: list of item IDs in ranked order
    relevant_with_scores: dict {item_id: relevance_score}
    """
    # Get relevance for recommended items
    gains = [relevant_with_scores.get(item, 0) for item in recommended[:k]]
    
    # Ideal: sort by relevance score
    ideal_gains = sorted(relevant_with_scores.values(), reverse=True)[:k]
    
    dcg = dcg_at_k(gains, k)
    idcg = dcg_at_k(ideal_gains, k)
    
    return dcg / (idcg + 1e-8)
```

### 10.5 MAP (Mean Average Precision)

Average precision for a single query, averaged over all users.

**AP (Average Precision) for one user:**
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

Simple and easy to interpret. Common in industry.

### 10.7 MRR (Mean Reciprocal Rank)

Where does the first relevant item appear?

```
MRR = (1/|U|) · Σ_u (1 / rank_of_first_relevant_item(u))
```

Great for "did the right answer appear near the top?" scenarios. Less appropriate when there are many relevant items.

### 10.8 Coverage and Catalog Utilization

Beyond accuracy metrics:

```
Catalog coverage = |unique items recommended| / |total items|
User coverage = |users receiving personalized recs| / |total users|
```

### 10.9 Business Metrics (the ones that actually matter)

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

**The correlation problem:** Offline NDCG improvements don't always translate to online CTR improvements. Always validate promising offline results with A/B tests.

---

## 11. Production Patterns

### 11.1 The Recommendation Pipeline

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

**Batch (offline) scoring:**
- Precompute recommendations for all users nightly
- Fast serving: lookup table
- Stale: can't incorporate recent behavior
- Good for: "weekly playlist" style features

**Near-real-time (streaming):**
- Update user embeddings as events stream in (Kafka + Flink)
- Candidate retrieval is fast (ANN lookup)
- Stale embeddings within a session: need to handle "you just watched X, don't recommend X again"

**Real-time scoring:**
- Full ranking model inference at request time
- Higher latency (typically 50-200ms budget for the entire pipeline)
- Most production systems use real-time retrieval + real-time ranking

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

Features need to be available both at training time and serving time, with the same values. A feature store solves this.

```
Data sources → Feature Store → Training pipeline
                     ↓
             Serving layer → Real-time inference
```

Key challenge: **training-serving skew**. If you compute a feature differently at training vs. serving time, your model is trained on data it will never see. Classic bugs:
- Time leakage: using future information at training time
- Different normalization
- Missing values handled differently

### 11.4 Caching

Item embeddings: precomputed, cached in memory (FAISS index). Refresh daily or weekly.

User embeddings: more dynamic. Options:
1. Precompute hourly (batch) — stale but fast
2. Compute on-demand with caching (1-hour TTL)
3. Streaming update as events arrive

Recommendation results:
- Cache per (user_id, context) with short TTL (5-15 minutes)
- Saves computation for repeated requests in a session
- Must invalidate on significant new interactions

### 11.5 A/B Testing

**The basics:**
- Split users randomly into control (A) and treatment (B) groups
- A sees old algorithm, B sees new algorithm
- Measure business metrics over a defined period (typically 2-4 weeks)
- Decide based on statistical significance

**Common pitfalls:**
- **Network effects:** If users interact with each other, A/B contamination occurs (social platforms)
- **Novelty effect:** New recommendations get inflated engagement just because they're new (usually fades in 2-4 weeks)
- **Multiple testing:** Running 10 experiments simultaneously inflates false positive rate → use Bonferroni correction or sequential testing
- **Carryover effects:** User behavior changes persist after experiment ends

**Interleaving:** Instead of two groups, interleave A and B results in a single ranked list and measure which items get clicked. Faster results (more sensitive), no user-split contamination risk. Used heavily at Netflix.

### 11.6 Monitoring

Key things to monitor in production:

```python
# Coverage — are we recommending diverse items?
coverage = len(unique_recommended_items_today) / total_catalog_size

# Popularity bias — are we over-recommending popular items?
popularity_of_recommended = mean(item_popularity for item in recommendations)

# Freshness — are we surfacing new content?
age_of_recommended = mean(days_since_published for item in recommendations)

# Model performance — CTR, conversion, watch time
# Track by user segment (new users, power users, etc.)
```

**Drift detection:** User behavior changes. Model degrades. Trigger retraining when:
- CTR drops more than 5% vs. 7-day moving average
- NDCG on held-out set drops below threshold
- Feature distribution shifts significantly (PSI/KL divergence monitoring)

---

## 12. Graph Neural Networks for Recommendations

### 12.1 Why Graphs?

The user-item interaction matrix is naturally a bipartite graph. Collaborative filtering implicitly uses 1-hop information (direct user-item interactions). But the real signal lives in multi-hop paths:

- User A liked Item X → Item X was also liked by User B → User B liked Item Y → recommend Item Y to User A

This is "second-order collaborative filtering" — friends of friends' taste. Graphs make this multi-hop reasoning explicit and learnable.

### 12.2 Graph Construction

```
Bipartite graph G = (U ∪ I, E)
- Nodes: users + items
- Edges: user-item interactions (weighted by interaction strength)

Can also be augmented:
- Item-item edges (same category, bought together)
- User-user edges (social connections)
- Attribute nodes (genre, brand)
```

### 12.3 Graph Convolutional Networks (GCN) for RS

**LightGCN (He et al., 2020)** — simplified GCN that removes non-linear transformations. Just propagates embeddings over the graph.

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
        
        # Trainable initial embeddings
        self.user_embed = nn.Embedding(n_users, embed_dim)
        self.item_embed = nn.Embedding(n_items, embed_dim)
        nn.init.normal_(self.user_embed.weight, std=0.01)
        nn.init.normal_(self.item_embed.weight, std=0.01)
    
    def forward(self, adj_matrix):
        # adj_matrix: normalized adjacency of bipartite graph
        
        all_user_embs = [self.user_embed.weight]
        all_item_embs = [self.item_embed.weight]
        
        user_emb = self.user_embed.weight
        item_emb = self.item_embed.weight
        
        for _ in range(self.n_layers):
            # Propagate: users aggregate from items, items aggregate from users
            new_user_emb = torch.sparse.mm(adj_matrix['UI'], item_emb)
            new_item_emb = torch.sparse.mm(adj_matrix['IU'], user_emb)
            user_emb = new_user_emb
            item_emb = new_item_emb
            all_user_embs.append(user_emb)
            all_item_embs.append(item_emb)
        
        # Layer aggregation
        final_user = torch.stack(all_user_embs, dim=1).mean(dim=1)
        final_item = torch.stack(all_item_embs, dim=1).mean(dim=1)
        
        return final_user, final_item
    
    def predict(self, user_ids, item_ids, adj_matrix):
        user_embs, item_embs = self.forward(adj_matrix)
        return torch.sum(user_embs[user_ids] * item_embs[item_ids], dim=-1)
```

### 12.4 PinSage (Pinterest)

Pinterest's GNN for billion-scale recommendations. Key innovations:
1. **Random walk sampling:** Full graph doesn't fit in memory. Sample local neighborhoods via random walks.
2. **Importance pooling:** Weight neighbor contributions by how often they appeared in random walks.
3. **Curriculum training:** Start with easy negatives, graduate to harder ones.
4. **MapReduce inference:** Compute embeddings in parallel across the graph.

### 12.5 Knowledge Graphs

Augment the bipartite user-item graph with a knowledge graph (item attributes, entity relationships):

```
Item "Inception" → directed_by → "Christopher Nolan"
"Christopher Nolan" → also_directed → "The Dark Knight"
"The Dark Knight" → genre → "Action"
```

Systems like KGCN, KGNN-LS, and RippleNet propagate user preferences through the KG, enabling more interpretable and generalizable recommendations.

### 12.6 When to Use GNNs vs. Standard CF

| | Standard CF | GNN |
|---|-------------|-----|
| Data scale | Works well | Scales with tricks (PinSage) |
| Multi-hop reasoning | Implicit (limited) | Explicit, controllable |
| Side information | Hard to incorporate | Natural (add attribute nodes) |
| Interpretability | Low | Can trace graph paths |
| Engineering complexity | Lower | Higher |
| Best for | Dense interactions | Sparse + rich graph structure |

---

## 13. Session-Based Recommendations

### 13.1 The Session Context

Sometimes you don't have a rich user history — or more importantly, you don't need it. What matters is what the user has done in the *current session*.

**Scenarios:**
- Anonymous user (no login)
- First session for a new user
- User in a specific temporary mood (browsing for a gift vs. for themselves)
- E-commerce: user is mid-shopping trip, context matters more than life history

"You've been looking at running shoes for the last 10 minutes. We recommend these socks and this GPS watch" — this is session-based, and it's correct even if your long-term profile shows you prefer dress shoes.

### 13.2 Markov Chain Models

Simple baseline. Assume the next item depends only on the last item (first-order Markov).

```
P(next item = j | last item = i) = count(i→j) / count(i)
```

Easy to implement, interpretable, but ignores non-adjacent items in the session.

### 13.3 RNN-Based Session Models

Treat the session as a sequence. Use GRU or LSTM to encode the session history into a hidden state, then predict the next item.

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
        
        # Use last hidden state for next-item prediction
        last_hidden = out[:, -1, :]  # (batch, hidden_dim)
        logits = self.output_layer(last_hidden)  # (batch, n_items)
        return logits
```

**Training:** Next-item prediction with cross-entropy loss. Treat each (session_prefix, next_item) pair as a training example.

### 13.4 Attention-Based Models: SASRec and BERT4Rec

**SASRec (Kang & McAuley, 2018):** Self-attentive sequential recommendation. Apply transformer self-attention to the sequence of items in a session.

Key advantage: can attend to any item in the session, not just recent ones (unlike GRU). Captures long-range dependencies in session behavior.

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
        
        # Causal mask: can only attend to past positions
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len)
        
        out = self.transformer(x, mask=mask)  # (batch, seq, embed_dim)
        last = out[:, -1, :]  # Last position
        return self.output_layer(last)
```

**BERT4Rec:** Applies BERT-style masked language modeling to item sequences. Mask random items in the sequence and train the model to predict the masked items. More data-efficient than next-item prediction.

### 13.5 Graph-Based Session Models

**SR-GNN (Wu et al., 2019):** Model each session as a directed graph, apply GNN to capture complex item transitions, then predict the next item using an attention mechanism that combines the session graph representation with the last item.

### 13.6 Hybrid: Combining Session with Long-Term History

**The key question:** When does the session override long-term history, and when does long-term history override the session?

Common approach: gated combination.

```
final_representation = α · session_encoding + (1 - α) · long_term_encoding

where α is learned from context (session length, recency, etc.)
```

If α ≈ 1: current session dominates (good for early in session, or for context-specific browsing)
If α ≈ 0: long-term history dominates (good for "what to watch tonight" style queries)

---

## 14. Common Interview Questions with Answers

### Q1: How would you design a recommendation system for Netflix from scratch?

**Answer:**

Start with requirements: ~250M users, ~15K titles, optimize for watch time and retention.

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

**CF:** Uses interaction patterns. "Users who behaved like you liked X." No need to understand what items are, just how people interact with them. Requires interaction history — fails for cold items and cold users.

**CB:** Uses item features. "Items similar to what you've liked." Works for cold items (just need features). Tends to over-specialize — can't discover items outside known taste.

**Use CF when:** Dense interaction data, you want serendipity, item features are hard to engineer (e.g., complex multimedia).

**Use CB when:** New item catalog, explainability is required ("because you liked genre X"), sparse user data.

**In practice:** Use both. Hybrid systems dominate production.

---

### Q3: What is matrix factorization and why does it work?

**Answer:**

MF decomposes the user-item interaction matrix R into two lower-dimensional matrices: P (users × k) and Q (items × k), such that R ≈ PQ^T.

The k latent dimensions capture hidden factors that explain user preferences — things like "affinity for action movies," "preference for indie content," etc. No one labels these; the model discovers them from observed interactions.

It works because the interaction matrix has low-rank structure: most user behavior can be explained by a small number of taste dimensions. 100M users might have complex individual histories, but their patterns of taste cluster into maybe 100-200 meaningful dimensions.

Optimization: minimize ||R - PQ^T||² over observed entries, with L2 regularization to prevent overfitting. Solved with SGD or ALS.

Key practical details: add bias terms (b_u, b_i), use k=64 to 512, regularize carefully, and for implicit feedback use confidence-weighted loss (iALS).

---

### Q4: How do two-tower models work? What makes them suitable for large-scale retrieval?

**Answer:**

Two-tower models have a separate neural network for users and items. Both towers output a fixed-dimensional embedding. The compatibility score is computed as a dot product (or cosine similarity) between the user and item embeddings.

The critical property: **the user and item towers never interact except at the final dot product.** This means:
- All item embeddings can be precomputed offline
- At query time, compute only the user embedding (one forward pass)
- Use approximate nearest neighbor (e.g., FAISS, ScaNN) to find the top-K items

This turns a problem that would require 100M forward passes into one forward pass + an ANN lookup. That's what makes it scale.

Training: in-batch negatives with cross-entropy loss (also called sampled softmax). Each item in the batch serves as a negative for all other users in the batch.

Challenge: popularity bias from in-batch negatives. Fix: subtract log(p_i) from item scores.

---

### Q5: Explain the cold start problem and how you'd address it.

**Answer:**

Cold start occurs when a user or item has no (or insufficient) interaction history for collaborative filtering to work.

**New user:**
- Onboarding survey (pick 3-5 preferences)
- Demographic-based proxies (age, location, device)
- Popular items fallback (safe but not personalized)
- Exploration slate: first session shows diverse content to quickly learn taste
- Cross-platform transfer (import ratings/history if available)

**New item:**
- Content-based features → embed in item space immediately
- Similar item lookup: find nearest established items by features, inherit their user engagement
- Warm-up injection: show new items to a random fraction of users with varied tastes, collect data, graduate to CF
- Feature-based two-tower: item tower takes features (not ID), so new items get an embedding from day 1

The deeper fix: use item feature representations (not just IDs) in your model so new items are never completely cold.

---

### Q6: How would you evaluate a recommender system? What's the difference between offline and online evaluation?

**Answer:**

**Offline metrics:** Computed on historical held-out data.
- Precision@K, Recall@K: coverage of relevant items in top-K
- NDCG@K: quality of ranking, position-discounted
- MAP: average precision across all users
- MRR: rank of first relevant item
- Hit Rate@K: binary, did we hit any relevant item?

**Limitation:** These are proxy metrics. Good NDCG doesn't guarantee good watch time.

**Online (A/B test):** Split users into control/treatment, measure real business metrics (CTR, watch time, retention). The ground truth for launch decisions.

**Why both?**
- Offline: cheap, fast, for development
- Online: expensive, slow, but measures what actually matters

**Common trap:** Offline gains don't always translate to online gains. Always A/B test before launch. And watch for novelty effects in A/B tests (inflate engagement short-term).

For retrieval specifically, use Recall@K (did the right items make it into the candidate set?) — this must be high or downstream ranking can't help.

---

### Q7: How do you handle popularity bias in recommendations?

**Answer:**

Popularity bias: popular items get recommended more → more interactions → ranked even higher → feedback loop. Long-tail items get no exposure.

**Causes:**
- Training data naturally has more signal for popular items
- In-batch negatives in two-tower training over-represents popular items as negatives → model learns to over-penalize popularity
- Ranking features like "total plays" discriminate against new items

**Fixes:**
- **Inverse propensity scoring (IPS):** Weight each training example by 1/p(item seen), where p depends on item popularity. Debiases the training objective.
- **Sampling correction in two-tower:** Subtract log(popularity) from item scores
- **Exploration slots:** Reserve fraction of recommended positions for long-tail items
- **Diversity constraints:** Cap the fraction of "popular" items in any slate
- **Separate popularity features from personalization features** in ranking model, so the model can learn "this user specifically likes obscure items"

---

### Q8: What is LambdaRank and why is it better than pointwise ranking?

**Answer:**

Pointwise ranking treats each (user, item) pair independently. It minimizes MSE or BCE on individual relevance scores. The problem: it doesn't care about the relative order of items, only about how well it predicts each score individually.

LambdaRank directly optimizes the *gradient* of the ranking metric (NDCG), even though NDCG itself isn't differentiable.

Key insight: you can define what the gradient *should be* without defining a differentiable loss. For a pair of items (i, j) where i should rank above j:

```
λ_ij = ∂L/∂(s_i - s_j) = -σ/(1 + e^{σ(si-sj)}) · |ΔNDCG_{ij}|
```

The |ΔNDCG| term scales the gradient by how important the swap would be. Swapping items at ranks 1 and 2 matters more than ranks 50 and 51.

**LambdaMART** = LambdaRank + gradient boosted trees. More interpretable, robust to feature engineering, widely used in production.

**Why better:** Directly optimizes what you care about (ranking quality at the top), not a surrogate that loosely correlates with it.

---

### Q9: How do you design for diversity in recommendations?

**Answer:**

Pure relevance optimization leads to a homogeneous list — 10 action movies when the user likes action movies. Users report satisfaction is higher with a diverse list.

**Approaches:**

1. **MMR (Maximal Marginal Relevance):** Greedily select items that maximize λ·relevance - (1-λ)·max_similarity_to_selected. λ controls the relevance-diversity tradeoff.

2. **Determinantal Point Processes (DPP):** Sample a set of K items from a distribution that rewards both quality and diversity (set of items with high quality AND low mutual similarity). Computationally expensive but principled.

3. **Calibration:** Ensure the distribution of recommended content matches the user's historical consumption distribution across categories. KL divergence penalty.

4. **Hard constraints:** Business rules — no more than 3 items of the same genre in a 10-item slate.

5. **Re-ranking layer:** Add diversity as a post-processing step after the ranking model, so ranking can focus on quality and re-ranking can enforce diversity.

The λ/tradeoff parameter is typically tuned via A/B testing, not held-out metrics.

---

### Q10: Walk me through how YouTube's recommendation system works.

**Answer:**

Based on Covington et al. (2016) and subsequent papers:

**Two-stage system:**

**Stage 1 — Candidate Generation (Two-Tower):**
- User tower: average embedding of watch history + search tokens + demographics + context (time of day, device)
- Item tower: video features, title, tags
- Trained to predict next video watch
- ANN retrieval fetches ~200 candidates from a corpus of millions

**Stage 2 — Ranking:**
- Richer features: user-video pair features not feasible at retrieval scale
- Video age (explicit feature to prevent "age bias" where model falsely correlates old videos with good quality)
- Video impressions (have we shown this to this user before? If yes, down-rank)
- Weighted logistic regression / deep neural network
- Multiple prediction heads: click, watch time, likes — multi-objective optimization

**Key design choices:**
- Predict *watch time*, not *clicks* — avoids clickbait
- "Example age" feature: model must explicitly learn freshness rather than confounding it with quality
- Asymmetric co-watch context: treat the most recent watch as label, prior watches as context → models "what next" rather than "what together"

---

### Q11: How would you handle a recommender system that has been deployed but performance is degrading?

**Answer:**

**Diagnose first:**

1. **Look at the metrics:** CTR degrading? NDCG? Coverage? Something specific to a user segment?
2. **Check for data drift:** Has the input feature distribution shifted? (PSI monitoring). New item types? Changed user behavior patterns?
3. **Check for label/feedback shift:** Is the click-through ratio on organic traffic changing? Has the product changed (new UI)?
4. **Check the data pipeline:** Stale features in the feature store? Missing features returning defaults? Training data cutoff too old?

**Common causes:**
- **Concept drift:** User taste and trends change. A model trained in January doesn't know about trending content in August.
- **Feedback loop:** Recommendations influence what users see → what users interact with → what the model trains on → creates narrow distribution collapse
- **Data quality issues:** Missing features, pipeline bugs, upstream schema changes

**Remediation:**
- Retrain on fresh data (schedule regular retraining — daily or weekly for fast-moving domains)
- Monitor feature importance: if important features went missing, fix the pipeline
- Add freshness features explicitly (item age, trending score)
- Evaluate training data distribution: are all user segments represented?

---

### Q12: BPR loss — write it out and explain the intuition.

**Answer:**

**BPR (Bayesian Personalized Ranking):**

For each user u, a sampled positive item i (interacted with), and a sampled negative item j (not interacted with):

```
L_BPR = -Σ_{(u,i,j)} log σ(r̂_ui - r̂_uj) + λ||Θ||²
```

**Intuition:** We want the score for the positive item to be higher than the score for the negative item. σ(r̂_ui - r̂_uj) is the probability of ranking i above j. We want to maximize this probability → minimize its negative log.

**Why better than pointwise BCE:**
- We don't assume we know the "true" relevance of an item — just that i is more relevant than j for user u
- This is a weaker and more reasonable assumption from implicit feedback
- The model is explicitly trained to *rank* correctly, not to predict absolute scores

```python
def bpr_loss(model, user_ids, pos_item_ids, neg_item_ids, reg=0.01):
    pos_scores = model(user_ids, pos_item_ids)
    neg_scores = model(user_ids, neg_item_ids)
    
    loss = -torch.log(torch.sigmoid(pos_scores - neg_scores)).mean()
    
    # L2 regularization
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

---

*This document covers the core concepts for ML interviews on recommender systems. The field moves fast — GNNs, large language models for recommendations (LLM4Rec), and retrieval-augmented recommendations are active research areas worth tracking.*
