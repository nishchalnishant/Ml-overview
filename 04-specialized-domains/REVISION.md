---
module: Specialized Domains
topic: Revision Card
subtopic: ""
status: unread
tags: [rl, recsys, gnn, revision, cheatsheet]
---
# Specialized Domains — 10-Minute Revision Card

Three domains. One card. RL → RecSys → GNN.

---

## Part 1: Reinforcement Learning

### Mental Model

Agent takes actions → environment returns reward → agent updates policy to maximize cumulative reward. No labeled examples; only outcome signals.

**When RL fits:** game playing, robotics, LLM alignment (RLHF). **When it doesn't:** you have labels → use supervised learning.

---

### MDP in 30 Seconds

$$\text{MDP} = (S, A, P, R, \gamma) \qquad \text{Return } G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k+1}$$

**Markov property:** future depends only on current state, not history.

**Bellman optimality:**
$$Q^*(s,a) = \mathbb{E}\left[r + \gamma \max_{a'} Q^*(s',a') \mid s,a\right]$$

---

### Algorithm Selector

| Situation | Use |
|-----------|-----|
| Discrete actions, large state | DQN |
| Continuous actions, stability matters | PPO |
| Continuous actions, sample efficiency | SAC |
| LLM alignment | PPO + RLHF or DPO |
| Small tabular MDP | Q-learning |
| Bandit (no state transitions) | UCB / Thompson Sampling |

---

### DQN Essentials

**Two innovations that make it work:**

1. **Experience replay** — random mini-batches from buffer break temporal correlation
2. **Target network** — frozen copy $\theta^-$ updated every N steps; prevents chasing a moving target

$$\mathcal{L}(\theta) = \mathbb{E}\!\left[\left(r + \gamma \max_{a'} Q(s',a';\theta^-) - Q(s,a;\theta)\right)^2\right]$$

**Gotcha:** DQN only works for discrete actions. For continuous → PPO or SAC.

---

### Policy Gradient Essentials

**The theorem:**
$$\nabla_\theta J(\theta) = \mathbb{E}\!\left[\sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot A_t\right]$$

Increase log-prob of actions with positive advantage; decrease for negative.

**Advantage:** $A(s,a) = Q(s,a) - V(s)$ — how much better than average?

**GAE:** $\hat{A}_t^{\text{GAE}} = \sum_{l=0}^{\infty}(\gamma\lambda)^l \delta_{t+l}$, $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$. At $\lambda=0$: 1-step TD. At $\lambda=1$: MC. Standard PPO: $\lambda=0.95$.

---

### PPO Clipped Objective

$$\mathcal{L}^{\text{CLIP}} = \mathbb{E}_t\!\left[\min\!\left(r_t(\theta)\hat{A}_t,\ \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right]$$

$r_t(\theta) = \pi_\theta(a_t|s_t) / \pi_{\theta_\text{old}}(a_t|s_t)$ — importance ratio.

**Intuition:** if action was good, increase its prob — but not by more than $1+\epsilon$. Prevents catastrophic large updates.

**Why PPO is default for RLHF:** stable, on-policy, handles large token action spaces.

---

### RLHF vs DPO

| | RLHF | DPO |
|--|------|-----|
| Reward model | Explicit (trained separately) | Implicit (in policy ratio) |
| RL loop | Yes (PPO) | No |
| Complexity | High | Low |
| Failure mode | Reward hacking | Distribution shift from offline data |

**DPO loss:**
$$\mathcal{L}_{\text{DPO}} = -\log\sigma\!\left(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_\text{ref}(y_w|x)} - \beta\log\frac{\pi_\theta(y_l|x)}{\pi_\text{ref}(y_l|x)}\right)$$

---

### RL Interview Quick-Draws

**"On-policy vs off-policy?"**
→ On-policy (PPO, A2C): learn about the current policy — data gets stale when policy changes. Off-policy (DQN, SAC): can reuse old data — enables experience replay.

**"Why does DQN need a target network?"**
→ Without it, both prediction and target move simultaneously — chasing a moving target → training oscillates. Frozen $\theta^-$ stabilizes the target for N steps.

**"What is the credit assignment problem?"**
→ With delayed sparse rewards, which of the 40 moves in a chess game caused the win? Solutions: discounting, eligibility traces, advantage estimation.

**"Reward hacking?"**
→ Agent finds unintended behaviors that score high on the reward proxy but not on true intent. RLHF example: verbose but unhelpful responses score high. Fix: KL penalty vs reference policy.

---

## Part 2: Recommender Systems

### Mental Model

Interaction matrix $R$ (users × items) is 99%+ sparse. Goal: fill in blanks. Two signals: **behavioral** (CF — who liked what) and **content** (CB — what items are like).

**Production pipeline:**
```
All items (100M) → [Two-Tower ANN] → 500 candidates
  → [Ranking model (DNN/LightGBM)] → Top 50
  → [Re-ranking: diversity, rules] → Final 10-20
```

---

### Algorithm Selector

| Situation | Use |
|-----------|-----|
| Dense interaction data | Matrix Factorization (ALS) |
| Large-scale retrieval | Two-Tower + ANN (FAISS) |
| Ranking over candidates | LambdaMART / DNN ranker |
| Cold new item | Content-based (feature tower) |
| Sequential / session | SASRec or GRU4Rec |
| Graph-structured interactions | LightGCN |

---

### Matrix Factorization

$$\hat{r}(u,i) = \mu + b_u + b_i + p_u^\top q_i$$

Minimize over observed entries: $\sum_{(u,i)\text{ obs.}} (\hat{r}_{ui} - r_{ui})^2 + \lambda(\|p_u\|^2 + \|q_i\|^2)$

**Implicit feedback (iALS):** confidence $c_{ui} = 1 + \alpha f_{ui}$. All (user, item) pairs used; zero-interactions get low confidence.

**ALS:** fix $Q$ → closed-form solve $P$; fix $P$ → solve $Q$. Parallelizable, no learning rate.

---

### Two-Tower Model

```
User features → User Tower → user_emb (d-dim)
                                    ↘ dot product → score
Item features → Item Tower → item_emb (d-dim)
```

**Why it scales:** item embeddings precomputed offline. At query time: one user forward pass + ANN lookup = milliseconds over 100M items.

**Training:** in-batch negatives — other items in the batch serve as negatives.

$$\mathcal{L} = \text{CrossEntropy}\!\left(\frac{\text{logits}}{\tau},\ \text{diagonal labels}\right)$$

**Sampling bias fix:** subtract $\log(p_i)$ from item scores where $p_i$ = sampling probability of item $i$.

---

### Ranking Paradigms

| Approach | Loss | Optimizes |
|----------|------|-----------|
| Pointwise | MSE/BCE per item | Absolute relevance |
| Pairwise (BPR) | $-\log\sigma(\hat{r}_{ui} - \hat{r}_{uj})$ | Relative order |
| Listwise (LambdaMART) | $\lambda_{ij} \propto \|\Delta\text{NDCG}\|$ | Full list quality |

**BPR gotcha:** treats all pairs equally — swapping ranks 1↔2 same weight as 50↔51. LambdaRank fixes this with $|\Delta\text{NDCG}|$ weighting.

---

### Evaluation Metrics Fast Reference

| Metric | Formula | Gotcha |
|--------|---------|--------|
| Precision@K | $\|\text{rel} \cap \text{top-K}\| / K$ | Doesn't penalize missing relevant items |
| Recall@K | $\|\text{rel} \cap \text{top-K}\| / \|\text{rel}\|$ | Treats all positions equally |
| NDCG@K | $\text{DCG} / \text{IDCG}$, DCG $= \sum (2^{r_i}-1)/\log_2(i+1)$ | Best: position-weighted |
| MAP | Mean AP over users | Penalizes both rank and recall |
| HR@K | Fraction of users with ≥1 hit | Simple, good for sparse data |

**Gotcha:** offline NDCG improvements don't reliably translate to online CTR. Always A/B test.

---

### Cold Start

| Problem | Fix |
|---------|-----|
| New user | Onboarding survey, popular fallback, exploration slate |
| New item | Content-based feature tower (works day-1), warm-up injection |

**Feature-based item tower:** item tower takes features (not IDs) → new item gets embedding immediately from its content.

---

### RecSys Interview Quick-Draws

**"CF vs content-based?"**
→ CF: behavioral patterns, needs history, enables serendipity. CB: item features, works for cold items, overspecializes. Production: always hybrid.

**"How do two-tower models scale to 100M items?"**
→ User and item only interact at the dot product. Precompute all item embeddings offline. At serving: 1 forward pass + ANN lookup.

**"Popularity bias?"**
→ Popular items get more training signal → recommended more → even more popular. Fix: sampling correction $(-\log p_i)$, IPS weighting, diversity constraints, exploration slots.

**"How do you detect recommendation degradation?"**
→ Monitor CTR trend, NDCG on held-out set, feature drift (PSI), catalog coverage dropping.

---

## Part 3: Graph Neural Networks

### Mental Model

Node embeddings = function of node features + neighborhood features. Stack $k$ layers → see $k$-hop neighborhood. Constraint: must be **permutation invariant** (reordering neighbors doesn't change result).

**Core message passing:**
```
For each layer k, for each node v:
  messages  = aggregate({h_u^(k-1) : u ∈ N(v)})
  h_v^(k)   = update(h_v^(k-1), messages)
```

---

### Architecture Selector

| Situation | Use |
|-----------|-----|
| Node classification (transductive) | GCN |
| Node classification (inductive, new nodes) | GraphSAGE |
| Heterogeneous neighbor importance | GAT |
| Maximum expressiveness (1-WL) | GIN |
| Recommendation at scale | LightGCN |
| Billion-node graphs | PinSage (random walk sampling) |
| Molecular property prediction | MPNN / NNConv (edge features) |
| Knowledge graph completion | TransE / RotatE |

---

### GCN

$$H^{(k+1)} = \sigma\!\left(\tilde{A} H^{(k)} W^{(k)}\right)$$

$\tilde{A} = \tilde{D}^{-1/2}(A+I)\tilde{D}^{-1/2}$ — symmetric normalized adjacency with self-loops.

**Self-loops** (+I): include node's own features in update.  
**Normalization** ($D^{-1/2}$): prevent high-degree hubs from dominating.

**Gotcha:** GCN is transductive — can't generalize to new nodes.

---

### GraphSAGE vs GCN vs GAT

| | GCN | GraphSAGE | GAT |
|--|-----|-----------|-----|
| Inductive | No | Yes | Yes |
| Aggregation | Fixed normalized mean | Learned (mean/max/LSTM) | Learned attention |
| New nodes | Can't | Can | Can |
| Edge features | No | No | Partial |
| Memory | $O(|E|d)$ | $O(\text{sample} \times d)$ | $O(|E| \times \text{heads} \times d)$ |

**GAT attention:**
$$\alpha_{vu} = \frac{\exp(\text{LeakyReLU}(a^\top [Wh_v \| Wh_u]))}{\sum_{w \in N(v)} \exp(\ldots)}$$

GAT = transformer attention restricted to the adjacency structure.

---

### GIN — Maximum Expressiveness

GIN achieves the maximum discriminative power of any message-passing GNN (= 1-WL test):

$$h_v^{(k)} = \text{MLP}^{(k)}\!\left((1+\epsilon^{(k)}) \cdot h_v^{(k-1)} + \sum_{u \in N(v)} h_u^{(k-1)}\right)$$

**Why sum > mean > max:** sum preserves neighborhood size information — mean conflates a node with 2 neighbors [A,B] with a node with 100 neighbors having the same mean.

---

### Over-Smoothing — The Core Failure Mode

After too many layers, all node representations converge to the same vector (random walk mixing).

**Symptom:** accuracy peaks at 2–3 layers, drops with more.

**Fixes:**
- Residual connections: $h_v^{(k)} = h_v^{(k-1)} + \text{AGG}(\ldots)$
- Initial residual (APPNP): $h_v^{(k)} = \alpha h_v^{(0)} + (1-\alpha)\text{AGG}(\ldots)$
- JK-Net: aggregate across all layers, not just the last

---

### Knowledge Graph Embeddings

**TransE:** $h + r \approx t$, score $= -\|h+r-t\|$. Fails on 1-to-N relations and symmetric relations.

**RotatE:** $t = h \odot r$ in complex space, $|r_i|=1$. Handles symmetry, antisymmetry, inversion, composition.

| Relation pattern | TransE | RotatE |
|-----------------|--------|--------|
| Symmetric (A↔B) | ❌ | ✓ (rotate by π) |
| 1-to-N | ❌ | ✓ |
| Composition | Partial | ✓ |

---

### Scalability

| Technique | Idea | Tradeoff |
|-----------|------|---------|
| Neighbor sampling (GraphSAGE) | Fix $k$ neighbors per layer | Gradient variance |
| Cluster-GCN | Mini-batch by graph partition | Cross-cluster info lost |
| SIGN | Precompute $A^1X, A^2X$... offline; train MLP | No online graph traversal |
| PinSage | Random walk neighbor importance | Approximate, memory-efficient |

**Neighbor explosion:** $k$ layers, avg degree $d$ → $d^k$ nodes per training example. Sampling is mandatory at $k \geq 2$.

---

### GNN Interview Quick-Draws

**"GCN vs GraphSAGE?"**
→ GCN is transductive (bakes adjacency into a fixed matrix, can't handle new nodes). GraphSAGE learns an aggregation function that generalizes — sample neighbors, concatenate own state + aggregated neighbors, transform. Works on unseen nodes.

**"What is over-smoothing?"**
→ Repeated neighborhood averaging blends all representations toward the same vector. Fix: 2–3 layers max, residual connections, JK-Net.

**"Why is sum better than mean in GIN?"**
→ Mean loses neighborhood size — a node with 2 neighbors [A,B] and one with 100 neighbors with the same average look identical. Sum preserves size → GIN achieves 1-WL expressiveness.

**"How do GNNs handle billion-node graphs?"**
→ Neighbor sampling (GraphSAGE): fix neighborhood size. Cluster-GCN: mini-batch by partition. SIGN: precompute multi-hop features offline, train MLP.

**"GAT vs transformer attention?"**
→ Both compute query-key attention weights. GAT restricts attention to the graph adjacency (sparse neighbors only). Transformer attends all-to-all (dense). Same mechanism, different mask.
