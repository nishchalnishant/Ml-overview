# Machine Learning Design Interview

---

## The Interview Framework

**The problem**: candidates who jump straight to model architecture fail design interviews because they answer a different question than what's being asked. The interviewer wants to see systems thinking, not model selection.

**The core insight**: ML design interviews test whether you can translate a vague product goal into a complete, defensible system — not whether you know the latest architecture.

**The mechanics**: seven-step framework applied to every question:

```
1. Clarify requirements and scope
   - "Is this for cold-start users or established ones?"
   - "What's the latency budget — 50ms or 500ms?"
   - "What does success look like as a business metric?"

2. Frame as an ML problem
   - What is the label? How is it defined?
   - Pointwise score, pairwise preference, or sequence?

3. Data sources and collection
   - What logs exist? What implicit signals are available?

4. Feature engineering
   - User features, item features, context features, interaction features

5. Model architecture
   - Justify complexity vs latency vs interpretability

6. Training pipeline
   - Data splits, loss function, optimization

7. Evaluation and deployment
   - Offline metrics → online A/B test → monitoring
```

**What breaks**: this framework becomes a ritual if you recite it without tailoring. Interviewers notice when you give a generic "feature engineering: user features, item features..." without naming concrete signals for the specific product. Name real features for the real system.

---

## Recommendation Systems

### Candidate Generation

**The problem**: scoring all items with a full ranking model is computationally impossible. A Netflix-scale system has millions of items; a pointwise neural ranker takes ~1ms per item — that's 1000 seconds per request.

**The core insight**: recall and precision are separate problems. Candidate generation maximizes recall cheaply; ranking maximizes precision expensively on a small set.

**The mechanics**:

Content-based filtering: represent items by their attributes; find items similar to user's history.

```python
# Item similarity via TF-IDF on descriptions
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

tfidf = TfidfVectorizer()
item_matrix = tfidf.fit_transform(item_descriptions)
# For a target item i, retrieve k nearest neighbors
similarities = cosine_similarity(item_matrix[i], item_matrix).flatten()
top_k = similarities.argsort()[-100:][::-1]
```

Collaborative filtering: two-tower model learns user and item embeddings; retrieve via approximate nearest neighbor (FAISS, ScaNN).

```python
# Two-tower embedding architecture
class TwoTower(nn.Module):
    def __init__(self, user_dim, item_dim, embed_dim):
        super().__init__()
        self.user_tower = nn.Sequential(
            nn.Embedding(user_dim, 256),
            nn.Linear(256, embed_dim)
        )
        self.item_tower = nn.Sequential(
            nn.Embedding(item_dim, 256),
            nn.Linear(256, embed_dim)
        )

    def forward(self, user_ids, item_ids):
        u = self.user_tower(user_ids)
        v = self.item_tower(item_ids)
        return torch.dot(u, v)  # cosine similarity for retrieval
```

At serving time: pre-compute all item embeddings offline. Query embedding lookup + ANN search returns top-1000 candidates in <10ms.

**What breaks**: two-tower models optimize for cosine similarity between user and item, which trains on positive pairs but ignores hard negatives. Without in-batch negative sampling or hard negative mining, the model learns to separate obvious positives from obviously irrelevant items, but fails to distinguish good from great candidates.

---

### Ranking

**The problem**: after candidate generation returns 1000 items, you need to order them so the best items appear first. The candidate generator optimized for recall, not quality of ordering.

**The core insight**: ranking is a supervised learning problem on pairs or lists, not just classification.

**The mechanics**:

Pointwise: train a binary classifier (clicked vs not-clicked). Fast but ignores relative ordering.

```python
# Pointwise: binary cross-entropy
loss = F.binary_cross_entropy(predicted_prob, clicked_label)
```

Pairwise: for each (positive, negative) pair, train the model to score positive higher.

```python
# Pairwise: BPR (Bayesian Personalized Ranking) loss
pos_score = model(user, positive_item)
neg_score = model(user, negative_item)
loss = -torch.log(torch.sigmoid(pos_score - neg_score)).mean()
```

Listwise: optimize a list-level metric directly. LambdaRank backpropagates gradients from NDCG changes.

```python
# LambdaRank: weight gradients by delta-NDCG
# Gradient for item i: sum_{j != i} lambda_ij
# where lambda_ij = |ΔNDCG_ij| * sigmoid(s_j - s_i)
```

Wide & Deep architecture for ranking: wide component memorizes specific feature crosses (user x item history); deep component generalizes via dense embeddings.

```python
class WideAndDeep(nn.Module):
    def __init__(self):
        super().__init__()
        self.wide = nn.Linear(sparse_feature_dim, 1)
        self.deep = nn.Sequential(
            nn.Linear(dense_embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, sparse_features, dense_embeds):
        wide_out = self.wide(sparse_features)
        deep_out = self.deep(dense_embeds)
        return torch.sigmoid(wide_out + deep_out)
```

**What breaks**: ranking trained on logged data inherits position bias — items shown in position 1 have higher click rates not because they're better but because users click what they see first. A model trained on raw clicks will learn to rank items that were historically shown first, creating a feedback loop.

---

### Position Bias Correction

**The problem**: click labels are biased by position. Item shown in position 1 gets 3x more clicks than position 5 even if both are equally relevant. Training on raw clicks teaches the model to prefer items that were historically positioned high.

**The core insight**: a click is the product of two probabilities — the probability that the user saw the item (examination probability, which depends on position) and the probability that the user found it relevant (true relevance). Separate them.

**The mechanics**: Inverse Propensity Scoring (IPS) reweights each training example by the inverse of its examination probability.

```python
# Propensity score: estimated probability of examination at position k
# Can be estimated via randomization experiments (swap positions randomly)
propensity = {1: 1.0, 2: 0.7, 3: 0.5, 4: 0.35, 5: 0.25}

# IPS-weighted loss: upweight examples from low positions, downweight high positions
def ips_loss(predicted, clicked, position):
    p_k = propensity[position]
    weight = 1.0 / p_k
    return weight * F.binary_cross_entropy(predicted, clicked)
```

Two-stage approach: (1) estimate propensity scores from a position-randomization experiment; (2) use those scores to debias training labels.

**What breaks**: IPS has high variance when propensities are small. Items at position 10 with propensity 0.1 get weight 10 — a single misclick inflates loss 10x. Clipped IPS (cap weight at a maximum) reduces variance at the cost of introducing bias.

---

### Calibration

**The problem**: a model's raw output is a score, not a probability. The model might output 0.8 for every positive example, but the true click rate is 5%. Downstream systems that use these scores as probabilities (budget allocation, ranking by expected value) will be wrong.

**The core insight**: calibration is a post-hoc correction — keep the ranking intact, fix the scale.

**The mechanics**:

Platt scaling: fit a logistic regression on (model_score → true_label) using a held-out validation set.

```python
from sklearn.linear_model import LogisticRegression
calibrator = LogisticRegression()
calibrator.fit(model_scores_val.reshape(-1, 1), true_labels_val)
calibrated_probs = calibrator.predict_proba(model_scores_test.reshape(-1, 1))[:, 1]
```

Temperature scaling: divide logits by a learned temperature T before softmax.

```python
# T > 1 softens distribution (more uncertainty); T < 1 sharpens it
T = 1.5
calibrated_logit = raw_logit / T
calibrated_prob = torch.sigmoid(calibrated_logit)
```

Isotonic regression: non-parametric monotone calibration, more flexible than Platt but needs more data.

**What breaks**: calibration assumes the validation distribution matches the serving distribution. If there is significant data drift between training and deployment (different user cohorts, different time period), calibration can make things worse by mapping to the wrong scale.

---

### Exploration vs Exploitation

**The problem**: a purely exploit-first system recommends only what it already knows works. New items get no exposure, stale preferences get reinforced, and the system cannot learn that user preferences have changed.

**The core insight**: you cannot learn what you do not observe. Some fraction of traffic must be used to gather new information (exploration) rather than maximizing immediate reward (exploitation).

**The mechanics**:

epsilon-greedy: with probability epsilon, serve a random item; with probability 1-epsilon, serve the best-scoring item.

```python
import random

def recommend(model, candidate_items, epsilon=0.1):
    if random.random() < epsilon:
        return random.choice(candidate_items)  # explore
    else:
        scores = model.score(candidate_items)
        return candidate_items[scores.argmax()]  # exploit
```

Thompson Sampling: maintain a Beta distribution over click probability for each item; sample from the distribution; pick the item with the highest sample.

```python
import numpy as np
from scipy.stats import beta

class ThompsonSampler:
    def __init__(self, n_items):
        self.alpha = np.ones(n_items)   # successes + 1
        self.beta_params = np.ones(n_items)  # failures + 1

    def select(self):
        samples = beta.rvs(self.alpha, self.beta_params)
        return samples.argmax()

    def update(self, item_id, reward):
        if reward:
            self.alpha[item_id] += 1
        else:
            self.beta_params[item_id] += 1
```

Contextual bandits (LinUCB): use context (user features) to decide exploration. Items with uncertain rewards in a specific context get boosted.

**What breaks**: exploration hurts short-term metrics. A 10% epsilon-greedy policy will decrease average CTR by approximately 10% on exploration requests. Companies compromise with small epsilon (1-2%) and restrict exploration to lower-stakes positions (not position 1).

---

## Search System Design

### Keyword vs Semantic Search

**The problem**: BM25 keyword search fails for vocabulary mismatch — a query "machine learning researcher" misses documents about "deep learning scientists." Semantic search alone is slower and harder to tune for exact-match queries.

**The core insight**: exact-match and semantic relevance are complementary signals. Hybrid retrieval uses both and combines their scores.

**The mechanics**:

BM25 (inverted index):

```
score(q, d) = sum_{t in q} IDF(t) * (tf(t,d) * (k1+1)) / (tf(t,d) + k1*(1-b+b*|d|/avgdl))
k1 = 1.2, b = 0.75 (standard parameters)
```

Fast lookup via inverted index: O(|query terms| x |documents containing term|).

Semantic search (bi-encoder):

```python
# Pre-encode all documents offline
doc_embeddings = encoder.encode(all_documents)  # [N, 768]
faiss_index = faiss.IndexFlatIP(768)
faiss_index.add(doc_embeddings)

# At query time
query_embedding = encoder.encode(query)
_, top_k_ids = faiss_index.search(query_embedding, k=100)
```

Hybrid: retrieve top-1000 from BM25 and top-1000 from semantic; merge; rerank with a cross-encoder.

Cross-encoder reranking (slower but more accurate):

```python
# Cross-encoder scores each (query, document) pair jointly
# Cannot pre-compute — must run at query time
scores = cross_encoder.predict([(query, doc) for doc in top_200_candidates])
reranked = sorted(zip(top_200_candidates, scores), key=lambda x: -x[1])
```

**What breaks**: cross-encoders are 50-100x slower than bi-encoders. A 200ms latency budget cannot afford cross-encoder scoring on all candidates. Use bi-encoder for recall, cross-encoder only for top-20 reranking.

---

### Training-Serving Skew in Search

**The problem**: the model is trained on logged query-document pairs from users who searched in the old system. The new model serves different queries, at different times, with different feature distributions — performance degrades silently.

**The core insight**: training data is a biased sample of what the model will see in production. The bias comes from the previous system's ranking decisions (selection bias), temporal drift, and features recomputed differently at training vs serving time.

**The mechanics**: log the exact feature vector used for ranking at serve time. Use those logged features for training, not re-computed features.

```python
# Anti-pattern: re-compute features at training time from raw events
# Leads to inconsistencies if any feature logic changed

# Correct pattern: log the feature vector at serving time
serving_log = {
    "query_id": "q123",
    "doc_id": "d456",
    "features": feature_vector.tolist(),  # exact values used at serving
    "position": 3,
    "clicked": True,
    "timestamp": "2024-01-15T10:23:00Z"
}
```

**What breaks**: point-in-time consistency is harder than it sounds. User profile features (e.g., "user's last 30 clicks") change continuously. The feature used at serving time (30 clicks as of 10:23am) differs from what gets recomputed at training time (30 clicks as of now). Log the actual values; do not recompute them.

---

## Fraud Detection

### The Core Challenge

**The problem**: fraud is rare (0.1-1% of transactions), adversarial (fraudsters adapt to detection), and expensive in both directions — false positives block legitimate transactions and false negatives let fraud through. Standard accuracy is meaningless on a 99% majority class.

**The core insight**: use PR-AUC (not ROC-AUC, not accuracy) because it directly measures precision-recall tradeoff on the minority class. ROC-AUC is dominated by the majority class and can report 0.98 AUC while catching almost no fraud.

**The mechanics**:

Feature engineering for fraud:

```python
fraud_features = {
    # Transaction features
    "amount": transaction.amount,
    "amount_vs_user_avg": transaction.amount / user_stats.avg_transaction_amount,
    "time_since_last_transaction_seconds": delta_t,

    # Velocity features (computed in real-time)
    "transactions_last_1h": velocity_store.get(user_id, window="1h"),
    "distinct_merchants_last_24h": velocity_store.distinct(user_id, field="merchant", window="24h"),
    "failed_attempts_last_10min": velocity_store.get(user_id, field="failed", window="10m"),

    # Behavioral anomaly
    "new_device": int(device_id not in user_known_devices),
    "new_location": int(geo_distance(last_location, current_location) > 500),  # km
    "unusual_hour": int(hour not in user_active_hours)
}
```

Evaluation:

```python
from sklearn.metrics import average_precision_score, precision_recall_curve

pr_auc = average_precision_score(y_true, y_scores)

# Set threshold to meet business SLA
# e.g., block top 0.5% of transactions with highest fraud score
precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
```

**What breaks**: fraud systems have a feedback loop problem. When you block a transaction, you will never observe what the true outcome would have been. This creates survivorship bias — the training data only contains transactions that passed the filter, underrepresenting the fraud distribution the model should learn to catch.

---

### Graph-Based Fraud Detection

**The problem**: individual transaction features miss ring fraud — where multiple accounts share devices, email providers, or shipping addresses in patterns that only become visible in the relationship graph.

**The core insight**: fraud leaves structural traces in the relationship graph that are invisible to feature-based models looking at single transactions in isolation.

**The mechanics**:

Build a bipartite graph: nodes are entities (users, devices, IP addresses, credit cards, email domains); edges connect them when they share a transaction or attribute.

GraphSAGE aggregates neighborhood information:

```python
class GraphSAGELayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W_self = nn.Linear(in_dim, out_dim)
        self.W_neigh = nn.Linear(in_dim, out_dim)

    def forward(self, node_features, neighbor_features):
        agg_neigh = neighbor_features.mean(dim=0)  # mean pooling
        h = torch.relu(self.W_self(node_features) + self.W_neigh(agg_neigh))
        return F.normalize(h, p=2, dim=-1)
```

R-GCN handles multiple edge types (user->device, user->email, user->merchant):

```python
class RGCN(nn.Module):
    def __init__(self, n_relations, in_dim, out_dim):
        super().__init__()
        # One weight matrix per relation type
        self.W = nn.ModuleList([nn.Linear(in_dim, out_dim) for _ in range(n_relations)])

    def forward(self, node_features, adj_matrices):
        out = sum(self.W[r](adj_matrices[r] @ node_features)
                  for r in range(len(adj_matrices)))
        return torch.relu(out)
```

Real-time pipeline: events flow through Kafka -> feature computation -> GNN scoring -> decision engine. Static embeddings pre-computed nightly; incremental updates for new entities.

**What breaks**: graph-based fraud detection requires near-real-time graph updates — a new fraud ring can appear within minutes. Pre-computed static embeddings become stale. Online GNN inference on a live graph with millions of nodes requires specialized infrastructure (DGL, PyG with streaming updates) that most teams lack.

---

## Feed Ranking

### Multi-Objective Optimization

**The problem**: a single engagement metric (maximize clicks) produces clickbait. Optimizing only for likes produces low-quality reshares. The business wants multiple objectives balanced simultaneously, and no single metric captures all of them.

**The core insight**: train separate models for each objective; combine their outputs with learned or manually tuned weights at ranking time. Each model sees the appropriate training population.

**The mechanics**:

```python
# Separate calibrated probability estimates per action type
p_like = like_model.predict(features)
p_comment = comment_model.predict(features)
p_share = share_model.predict(features)
p_hide = hide_model.predict(features)  # negative signal

# Final score: weighted combination
# Weights reflect business priorities and relative action rates
final_score = (
    0.4 * p_like +
    0.3 * p_comment +
    0.3 * p_share -
    2.0 * p_hide  # penalize hide/report heavily
)
```

MMOE (Multi-gate Mixture of Experts): shared experts learn common representations; separate gates per task select which experts are relevant for that task.

```python
class MMoE(nn.Module):
    def __init__(self, n_experts=8, n_tasks=3, input_dim=256, expert_dim=128):
        super().__init__()
        self.experts = nn.ModuleList([nn.Linear(input_dim, expert_dim)
                                      for _ in range(n_experts)])
        self.gates = nn.ModuleList([nn.Linear(input_dim, n_experts)
                                    for _ in range(n_tasks)])
        self.task_heads = nn.ModuleList([nn.Linear(expert_dim, 1)
                                         for _ in range(n_tasks)])

    def forward(self, x):
        expert_outputs = torch.stack([e(x) for e in self.experts], dim=1)  # [B, n_experts, d]
        outputs = []
        for i in range(len(self.task_heads)):
            gate = torch.softmax(self.gates[i](x), dim=-1)  # [B, n_experts]
            mixed = (gate.unsqueeze(-1) * expert_outputs).sum(dim=1)  # [B, expert_dim]
            outputs.append(self.task_heads[i](mixed))
        return outputs
```

**What breaks**: weights in the final score function encode implicit value judgments. A weight that increases time-on-site may simultaneously increase anxiety. Multi-objective systems require explicit choices about which human outcomes to optimize that go beyond A/B test CTR metrics.

---

### Re-Ranking for Freshness and Diversity

**The problem**: a pure relevance ranker shows the same top-performing content repeatedly. Users see yesterday's viral post again; their feed looks identical across sessions; the system never surfaces emerging content.

**The core insight**: the optimal list maximizes user utility across the session, not item-level relevance. Diversity and freshness are list-level properties that cannot be optimized item-by-item.

**The mechanics**:

Freshness boost: decay content score over time.

```python
import math

def freshness_score(base_score, age_hours, half_life_hours=6):
    decay = math.exp(-0.693 * age_hours / half_life_hours)
    return base_score * decay
```

Maximal Marginal Relevance (MMR) for diversity: iteratively select items that maximize relevance minus similarity to already-selected items.

```python
def mmr_rerank(scored_items, item_embeddings, lambda_=0.5, k=20):
    selected = []
    remaining = list(range(len(scored_items)))

    for _ in range(k):
        if not selected:
            best = max(remaining, key=lambda i: scored_items[i].score)
        else:
            selected_embeds = torch.stack([item_embeddings[i] for i in selected])
            def mmr_score(i):
                relevance = scored_items[i].score
                max_sim = F.cosine_similarity(
                    item_embeddings[i].unsqueeze(0), selected_embeds
                ).max().item()
                return lambda_ * relevance - (1 - lambda_) * max_sim
            best = max(remaining, key=mmr_score)

        selected.append(best)
        remaining.remove(best)

    return [scored_items[i] for i in selected]
```

**What breaks**: MMR is O(k^2) in the number of selected items. At k=100 with 768-dim embeddings, the per-request cost becomes significant. Production systems approximate diversity with simpler rules: no two items from same author in top 5, no same topic cluster in adjacent positions.

---

## Ads Ranking

### GSP Auction Mechanics

**The problem**: a naive highest-bidder-wins auction incentivizes advertisers to bid their true value once, then defect. It also ignores ad quality — a high-bidding irrelevant ad harms user experience and reduces long-term platform revenue.

**The core insight**: charge the minimum price needed to hold the position, not the actual bid. Weight by quality to align ad relevance with revenue. This makes truthful bidding the dominant strategy.

**The mechanics**: Generalized Second Price (GSP) auction — each advertiser wins based on expected value, not raw bid:

```
EV_i = bid_i * pCTR_i * quality_i
```

```python
def gsp_auction(advertisers, ad_slots):
    """
    advertisers: list of {bid, predicted_ctr, quality_score}
    """
    for ad in advertisers:
        ad['ev'] = ad['bid'] * ad['predicted_ctr'] * ad['quality_score']

    ranked = sorted(advertisers, key=lambda x: -x['ev'])

    allocation = []
    for slot_idx, slot in enumerate(ad_slots):
        if slot_idx >= len(ranked):
            break
        winner = ranked[slot_idx]
        if slot_idx + 1 < len(ranked):
            next_ev = ranked[slot_idx + 1]['ev']
            # Second-price: pay minimum to hold position
            price_per_click = (next_ev / winner['predicted_ctr'] /
                               winner['quality_score']) + 0.01
        else:
            price_per_click = slot.reserve_price
        allocation.append({'advertiser': winner, 'price_per_click': price_per_click})

    return allocation
```

**What breaks**: GSP is not perfectly incentive-compatible — in theory, VCG achieves this, but GSP is computationally simpler and empirically stable. The real problem is that predicted CTR directly determines price, so a model that underpredicts CTR undercharges advertisers (lost revenue); a model that overpredicts overcharges and erodes advertiser trust.

---

### CTR and CVR Prediction (ESMM)

**The problem**: training a CVR (conversion rate) model directly on clicks introduces sample selection bias. CVR labels only exist for clicked items — but at serving time, the model scores all items, including those never clicked. The CVR model has never seen the distribution it is predicting on.

**The core insight**: decompose the joint probability using the probability chain rule. Train CTR and post-click CVR jointly, letting each model see the appropriate population.

**The mechanics**: ESMM (Entire Space Multi-Task Model) by Alibaba:

```
P(click AND convert) = P(click) * P(convert | click)
                       [CTR model]   [CVR model on click space]
```

```python
class ESMM(nn.Module):
    def __init__(self, input_dim, embed_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embed_dim)

        # CTR tower: trained on all impressions
        self.ctr_tower = nn.Sequential(
            nn.Linear(embed_dim, 256), nn.ReLU(),
            nn.Linear(256, 1), nn.Sigmoid()
        )

        # CVR tower: trained on click space, deployed on all impressions
        self.cvr_tower = nn.Sequential(
            nn.Linear(embed_dim, 256), nn.ReLU(),
            nn.Linear(256, 1), nn.Sigmoid()
        )

    def forward(self, features):
        embeds = self.embedding(features).mean(dim=1)
        p_ctr = self.ctr_tower(embeds)
        p_cvr = self.cvr_tower(embeds)
        p_ctcvr = p_ctr * p_cvr  # P(click AND convert)
        return p_ctr, p_cvr, p_ctcvr

def esmm_loss(p_ctr, p_cvr, p_ctcvr, click_labels, conversion_labels):
    ctr_loss = F.binary_cross_entropy(p_ctr, click_labels)
    # Conversion label is 1 only if both clicked and converted
    ctcvr_loss = F.binary_cross_entropy(p_ctcvr, click_labels * conversion_labels)
    return ctr_loss + ctcvr_loss
```

**What breaks**: ESMM assumes the CVR tower sees correct conversion rates for clicked items. If click fraud distorts the click population, CVR estimates become unreliable. Also, CVR for extremely rare events (0.01% conversion rate) still suffers from high variance — the model has seen very few positive conversion examples even in the click space.

---

### Calibration for Downsampling

**The problem**: for CTR prediction, positive examples (clicks) are rare — typically 1-5% of impressions. Training on the full dataset is wasteful; downsampling negatives is common. But downsampled models predict uncalibrated probabilities that are too high.

**The core insight**: downsampling changes the apparent base rate. You can analytically correct for this without refitting the model.

**The mechanics**:

```python
# Downsample negatives by factor w (keep 1/w of negatives)
# This multiplies the apparent positive rate by w

def correct_downsampled_prob(p_prime, w):
    """
    p_prime: model output trained on downsampled data
    w: downsampling factor (e.g., 100 means kept 1% of negatives)
    """
    # Derived from Bayes' theorem on shifted base rates
    p_true = p_prime / (p_prime + (1 - p_prime) / w)
    return p_true

# Example: model predicts 0.5, but data was 100x downsampled
# True CTR is ~0.005, not 0.5
true_ctr = correct_downsampled_prob(p_prime=0.5, w=100)
# true_ctr ~= 0.005
```

**What breaks**: this correction assumes the only difference between sampled and full data is the negative rate. If the positives were also filtered (e.g., only high-quality clicks kept), the correction does not apply. Calibration correction works for base rates but does not fix distributional drift in the feature space.

---

### Budget Pacing

**The problem**: an advertiser with a daily budget of $1000 should not exhaust it in the first hour when traffic is high. They would get no impressions for the remaining 23 hours, miss their target audience at different times, and have poor return on ad spend.

**The core insight**: smooth budget spending by introducing a throttle rate that controls what fraction of auction opportunities the advertiser participates in, adjusting based on actual vs target spend.

**The mechanics**:

```python
class BudgetPacer:
    def __init__(self, daily_budget):
        self.daily_budget = daily_budget
        self.spent = 0.0

    def throttle_rate(self, current_hour):
        elapsed_fraction = current_hour / 24.0
        target_spent = self.daily_budget * elapsed_fraction
        actual_spent = self.spent

        ratio = actual_spent / (target_spent + 1e-6)

        if ratio < 0.8:    # underspending: participate more
            return min(1.0, 1.2)
        elif ratio > 1.2:  # overspending: participate less
            return max(0.0, 0.6)
        else:
            return 1.0

    def should_participate(self, current_hour):
        rate = self.throttle_rate(current_hour)
        return random.random() < rate
```

**What breaks**: throttling randomly excludes the advertiser from auctions. During excluded auctions, a competitor might win a valuable placement. Smarter pacing uses bid shading (reduce bid when ahead of pace, increase when behind) instead of random exclusion, which preserves auction participation while controlling spend.

---

## Latency Constraints

**The problem**: production ML systems have hard latency budgets. An ads system must return a ranked list in <50ms from request receipt. A 200ms ranking model that achieves +2% CTR may actually reduce revenue because slower page loads reduce click rate by 5%.

**The core insight**: latency and quality are in direct tension. System design is about choosing where to spend the latency budget.

**The mechanics**: typical latency breakdown for ads:

```
Total budget: 50ms

- Feature retrieval (user profile, item features): 5ms
  [Redis lookup, pre-computed features]

- Candidate generation (ANN search): 5ms
  [FAISS, ScaNN -- return top-1000]

- Feature assembly for ranking: 3ms
  [join user features x candidate features]

- Ranking model inference (top-1000 -> top-50): 15ms
  [XGBoost or shallow MLP, batched]

- Re-ranking and business logic: 5ms
  [diversity, freshness, pacing filters]

- Response serialization and network: 10ms

Remaining buffer: 7ms
```

Optimization techniques:
- Model quantization (FP32 to INT8): 2-4x inference speedup with <1% accuracy loss
- ONNX export: portable format with optimized runtime
- Pre-compute item embeddings offline; only compute query embedding at serve time
- Cascade: use fast model to filter candidates; expensive model only on top-100

**What breaks**: optimizing for p50 latency misses tail latency. A 50ms p50 with 500ms p99 means 1% of users see 10x slower responses — often the most critical moments (first purchase, fraud decision) that trigger expensive feature lookups. Monitor p99 and p999 separately.

---

## Case Studies

### Airbnb Search Ranking

**The problem**: short-term rental search has unique challenges — inventory is heterogeneous, availability is scarce and time-constrained, and both host and guest preferences matter. Training on clicks undervalues listings that rarely get shown but convert at high rates.

**The core insight**: use booking confirmation as the training signal, not click. A clicked listing that leads to no booking provides no signal about quality; a non-clicked listing that a better system would have shown represents a missed booking.

**The mechanics**:
- Primary label: booking confirmation (binary, delayed by days)
- Feature engineering: listing features (reviews, photos, price/night, amenities), guest features (verification level, booking history, response rate), search context (check-in date, group size, destination)
- Listing quality score trained separately from ranking; both fed into a joint model
- Geo features: distance from destination center, proximity to attractions

**What breaks**: scarcity distorts signals. A high-quality listing always booked immediately appears in logs with few impressions but nearly 100% conversion. A mediocre listing always available shows up constantly with low conversion. Without accounting for supply constraints, the model learns "available = low quality" and deprioritizes scarce, desirable listings.

---

### TikTok Feed Ranking

**The problem**: video content consumption differs from text — completion rate and replays are stronger signals than likes, which require active effort. A standard engagement model optimized for likes produces clickbait thumbnails, not genuinely engaging content.

**The core insight**: video completion rate is the primary signal because it is passive and harder to game. A user who watches 95% of a video genuinely engaged with it.

**The mechanics**:
- Primary training signal: video completion rate = watched_seconds / video_duration
- Secondary signals: like, comment, share, follow (positive); skip, scroll past (negative)
- Two-tower model: user tower (watch history, follow graph, device, locale) x video tower (visual features from frame sampling, audio features, caption embeddings, hashtags)
- Cold-start for new videos: initial traffic allocation based on creator history and content features; boost if early viewers show high completion

**What breaks**: optimizing for completion rate creates filter bubbles faster than any other signal. A user who watches three videos on any topic will be shown more of the same, because the completion signal is very strong. The recommendation loop amplifies whatever the user engaged with, with no natural brake.

---

## Offline vs Online Metrics Alignment

**The problem**: models that improve offline metrics (AUC, NDCG, PR-AUC) often do not improve online business metrics (CTR, revenue, session length). Teams spend months optimizing a model that launches and shows no improvement.

**The core insight**: offline metrics measure model quality on historical data; online metrics measure user behavior in a live interactive system. The gap between them is real and must be closed empirically, not assumed away.

**The mechanics**: track the offline-to-online correlation history:

```
| Experiment | Offline delta-AUC | Online delta-CTR | Online delta-Revenue |
|------------|-------------------|------------------|----------------------|
| Exp-001    | +0.5%             | +0.3%            | +0.2%                |
| Exp-002    | +1.2%             | -0.1%            | -0.3%  <- offline improvement didn't transfer
| Exp-003    | +0.2%             | +0.8%            | +1.1%  <- small offline, large online gain
```

When offline and online diverge: investigate position bias in offline data, check if A/B test traffic is representative, verify that serving features match training features exactly.

**What breaks**: A/B tests have minimum detectable effect sizes. A 0.1% revenue improvement requires millions of users to detect at 95% confidence. Teams run underpowered experiments, see "no significant difference," and ship anyway — accumulating many small errors that compound over time.
