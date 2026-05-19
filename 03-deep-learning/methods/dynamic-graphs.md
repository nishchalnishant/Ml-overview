# Dynamic Graphs, Temporal GNNs, and Graph Generation

> Graphs that change over time and methods to model, predict, and generate graph-structured data.

---

## Table of Contents

1. [Dynamic Graphs Overview](#1-dynamic-graphs-overview)
2. [Discrete-Time Dynamic Graphs (DTDG)](#2-discrete-time-dynamic-graphs-dtdg)
3. [Continuous-Time Dynamic Graphs (CTDG)](#3-continuous-time-dynamic-graphs-ctdg)
4. [Temporal Link Prediction](#4-temporal-link-prediction)
5. [Graph Generation](#5-graph-generation)
6. [VGAE — Variational Graph Autoencoder](#6-vgae--variational-graph-autoencoder)
7. [Sequential Graph Generation](#7-sequential-graph-generation)
8. [Diffusion-Based Graph Generation](#8-diffusion-based-graph-generation)
9. [Molecule Generation](#9-molecule-generation)
10. [Evaluation Metrics](#10-evaluation-metrics)
11. [Key Interview Points](#11-key-interview-points)

---

## 1. Dynamic Graphs Overview

**The problem:** standard GNNs operate on a fixed, static graph. But most real systems evolve — friendships form and dissolve, transactions occur at specific times, traffic speeds change every minute, molecules change bond structure during simulation. A static GNN trained on a snapshot can't predict what happens next or understand how the graph reached its current state.

**The core insight:** model the graph's evolution explicitly. Two fundamentally different data structures arise depending on whether time is discretized:

| Paradigm | Representation | Clock | Typical data |
|---|---|---|---|
| Discrete-Time Dynamic Graph (DTDG) | Sequence of snapshots G_1, ..., G_T | Fixed interval | Citation networks, traffic grids |
| Continuous-Time Dynamic Graph (CTDG) | Ordered event stream (u, v, t, feat) | Real-valued timestamp | Social interactions, financial transactions |

**Applications:**
- Social networks: friend/follow links appear and disappear
- Financial transactions: detect fraud by modeling interaction bursts
- Traffic: road-segment speeds change every minute
- Molecular dynamics: atom bonds break and form during simulation
- Knowledge graphs: facts have validity intervals

**What breaks in all dynamic graph methods:** temporal dependency — future events must never inform past predictions. Standard random train/val/test splits leak future information into training. All temporal graph methods require strict time-based splits.

---

## 2. Discrete-Time Dynamic Graphs (DTDG)

**The problem:** you have a sequence of graph snapshots at equal time intervals. How do you model the state of a node at time T given its history of neighborhoods at times 1, ..., T-1?

**The core insight:** treat the sequence of graph snapshots like a sequence in a language model. Extract spatial features from each snapshot with a GNN; then feed the sequence of GNN outputs into a temporal model (RNN or Transformer). The two components handle different aspects of the data: GNN → local structural context, RNN/Transformer → temporal evolution.

### EvolveGCN

**The problem:** the standard DTDG recipe (GNN → RNN) assumes a fixed node set across snapshots. When nodes appear or disappear between snapshots, the GNN hidden state for new nodes is undefined, and old hidden states become stale.

**The core insight:** instead of carrying hidden state *per node* through time, carry the GNN *weight matrices* through time. The GCN weights at time t are the output of an RNN whose input is information about the previous snapshot. This decouples temporal modeling from node identity — even if nodes come and go, the weight matrices evolve smoothly.

**The mechanics:**
```
W_t = GRU(W_{t-1}, H_{t-1})    # weights are the RNN hidden state
H_t = GCN(A_t, X_t; W_t)
```

Two variants:
- **EvolveGCN-H:** uses the node embedding matrix H_{t-1} as input to the GRU — better when node features are informative.
- **EvolveGCN-O:** uses only W_{t-1} — handles graphs where the node set changes completely.

**What breaks:** the GRU that evolves weights sees a matrix input at each step; this is computationally expensive for large GCNs. EvolveGCN is practical only for moderate network widths.

### GCRN (Graph Convolutional RNN)

**The problem:** in the GNN → RNN pipeline, graph convolution and gating (RNN) happen at separate stages. Spatial graph structure influences temporal modeling only indirectly through the embedding passed to the RNN.

**The core insight:** replace the linear transformation inside each GRU gate with a graph convolution. Now spatial message-passing and temporal gating happen simultaneously within a single unified cell.

**The mechanics:**
```
Standard GRU gate:  z_t = sigmoid(W_z x_t + U_z h_{t-1})

GCRN gate:          z_t = sigmoid(GraphConv(A_t, x_t) + GraphConv(A_t, h_{t-1}))
```

**What breaks:** the graph structure A_t must be available at every recurrent step. For sparse, irregular graphs this is fine, but for dense graphs the graph convolution inside the gating becomes a bottleneck.

---

## 3. Continuous-Time Dynamic Graphs (CTDG)

**The problem:** real-world events don't happen at fixed intervals. A financial fraud might burst in milliseconds; a social connection might form once a year. Discretizing to snapshots either loses sub-interval resolution (coarse snapshots) or produces mostly empty snapshots (fine snapshots). You need a formulation that treats timestamps as first-class inputs.

**The core insight:** represent the graph as an ordered stream of timestamped events (u, v, t, features). To embed a node at query time t, aggregate its temporal neighborhood: all events involving u at times t' < t. Encode time differences directly into the embedding so the model knows *how long ago* each interaction occurred.

```
E = {(u_1, v_1, t_1, f_1), (u_2, v_2, t_2, f_2), ...}   t_1 ≤ t_2 ≤ ...

Temporal neighborhood of u at query time t:
N(u, t) = {(v, t', f) : (u, v, t') ∈ E, t' < t}
```

### TGAT (Temporal Graph Attention)

**The problem:** standard attention uses positional embeddings to encode order. For temporal graphs, position is a real-valued timestamp — you need to encode arbitrary time differences as a vector that can participate in attention without discretizing or binning time.

**The core insight:** Bochner's theorem says any stationary positive-definite kernel k(Δt) can be expressed as an expectation of random Fourier features. Use learnable Fourier features to map scalar time differences to vectors that act as time-aware positional encodings.

**The mechanics:**
```python
class TimeEncoding(nn.Module):
    def __init__(self, d_time):
        super().__init__()
        self.w = nn.Parameter(torch.randn(d_time // 2))
        self.b = nn.Parameter(torch.randn(d_time // 2))

    def forward(self, t):
        # t: (B,) float tensor of timestamps
        t = t.unsqueeze(-1)
        x = t * self.w + self.b
        return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)  # (B, d_time)
```

Using relative time `t − t'` (not absolute timestamp) improves generalization across different time ranges.

TGAT layer:
```
h_u(t) = Attn(Q=φ(u,t),  K=φ(v, t') for (v,t') in N(u,t),
               V=concat(feat_{uv}, TimeEnc(t − t')))
```

**What breaks:** the temporal neighborhood N(u, t) must be fetched at inference time — O(|N(u,t)|) lookups per node per forward pass. For high-degree nodes in dense graphs (e.g., celebrities in social networks), this becomes a bottleneck.

### TGN (Temporal Graph Networks)

**The problem:** TGAT recomputes a node's embedding from scratch every query time, using its temporal neighborhood. But a node's entire history is re-processed at every query. For nodes with long histories, this is expensive. More importantly, nodes that haven't been queried recently carry no summary of their past — they're treated as if they appeared fresh.

**The core insight:** give each node a persistent memory vector that summarizes its interaction history. When a new event involves node u, update u's memory using the event's information. When querying u's embedding, start from its current memory (a compressed representation of all past events) rather than re-processing the entire history.

**The mechanics:**
```
Memory:     s_u ∈ ℝ^d  — persistent per-node state, updated after each event
Messages:   m_u(t) = msg(s_u(t⁻), s_v(t⁻), Δt, e_{uv}(t))
Aggregation: ā_u(t) = AGG({m_u(t') : t' ≤ t})
Memory update: s_u(t) = GRU(ā_u(t), s_u(t⁻))
Embedding:  z_u(t) = GNN(s_u(t), N(u,t))  # temporal graph attention on top of memory
```

```python
class TGN(nn.Module):
    def __init__(self, num_nodes, mem_dim, time_dim, feat_dim):
        super().__init__()
        self.memory    = torch.zeros(num_nodes, mem_dim)
        self.time_enc  = TimeEncoding(time_dim)
        self.msg_fn    = nn.Linear(2 * mem_dim + time_dim + feat_dim, mem_dim)
        self.mem_update = nn.GRUCell(mem_dim, mem_dim)
        self.embedding  = TemporalGraphAttention(mem_dim, time_dim)

    def compute_messages(self, src, dst, t, edge_feat):
        dt  = self.time_enc(t)
        raw = torch.cat([self.memory[src], self.memory[dst], dt, edge_feat], dim=-1)
        return src, self.msg_fn(raw)

    def update_memory(self, node_ids, messages):
        self.memory[node_ids] = self.mem_update(messages, self.memory[node_ids])

    def forward(self, src, dst, t, edge_feat):
        node_ids, msgs = self.compute_messages(src, dst, t, edge_feat)
        self.update_memory(node_ids, msgs.detach())  # detach: no BPTT through full history
        z_src = self.embedding(src, t, self.memory)
        z_dst = self.embedding(dst, t, self.memory)
        return z_src, z_dst
```

**What breaks:** memory staleness — a node's memory is only updated when it participates in an event. Rarely-active nodes carry stale memories from months ago. TGN partially addresses this by updating both source and destination memories on each event, but nodes that are never involved in events have no memory updates at all.

---

## 4. Temporal Link Prediction

**The problem:** given all events up to time t, predict whether edge (u, v) will appear at time t' > t. This is the canonical evaluation task for temporal graph models.

### Evaluation protocol: time-based splits are mandatory

```
|──────────── train ────────────|── val ──|── test ──|
t=0                             t1        t2         t_max
```

Random splits cause temporal leakage — events from t > t1 appear in training data, giving the model access to future information. This produces inflated metrics that don't reflect real deployment performance.

| Setting | Nodes at test time | Challenge |
|---|---|---|
| Transductive | Seen during training | Standard generalization |
| Inductive | New nodes not in training | Model must generalize to unseen nodes |

TGN handles the inductive setting because memory is computed from event history, not a fixed embedding lookup. A new node with no history starts with zero memory and accumulates state from its first events.

### Scoring

```python
pos_score = (z_src * z_dst).sum(-1)           # dot product for positive edge
neg_score = (z_src * z_neg_dst).sum(-1)       # random negative destination
loss = F.binary_cross_entropy_with_logits(
    torch.cat([pos_score, neg_score]),
    torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)])
)
```

Metrics: Average Precision (AP), ROC-AUC, Mean Reciprocal Rank (MRR).

**What breaks:** random negatives are easy — any unconnected node pair qualifies. Models trained with random negatives look much better than they are. Historical negatives (edges that existed but no longer do) or structurally-similar non-edges are more realistic and harder.

---

## 5. Graph Generation

**The problem:** generating graphs is harder than generating images. Images have a fixed n×n grid — every pixel has a canonical position. Graphs have no canonical representation. The same graph can be represented by n! different adjacency matrices (one per node permutation). A generative model must be invariant to this permutation while still producing diverse, valid graphs.

Five compounding difficulties:

1. **Variable size:** unlike images, graphs have no fixed n×n grid.
2. **Permutation invariance:** the same graph has n! adjacency matrix representations — the model must handle all of them.
3. **Validity constraints:** chemical graphs require valid valencies, connectivity, no isolated atoms.
4. **Sparsity:** adjacency matrices are sparse; modeling all n² entries penalizes sparsity poorly and is wasteful.
5. **Evaluation:** no pixel-wise MSE analogue; need domain-specific metrics (chemical validity, graph statistics).

---

## 6. VGAE — Variational Graph Autoencoder

**The problem:** you want to learn a latent representation of a graph that captures its structure. Given a new set of node features, you want to predict which edges should exist. Standard autoencoders for tabular data don't handle relational structure — encoding must propagate information over edges, and decoding must predict edges.

**The core insight:** apply the VAE framework to graphs. Use a GCN as the encoder — it propagates node features over the observed edges to produce a latent embedding per node. Decode by checking whether pairs of nodes should be connected: nodes with similar latent vectors get an edge. The inner product of two latent vectors is the logit for the edge probability.

**The mechanics:**
```
Encoder:  Z = GCN(A, X) → μ, log σ  (reparameterize to z_i ∈ ℝ^d)
Decoder:  Â_{ij} = sigmoid(z_iᵀ z_j)

ELBO: L = E_q[log p(A|Z)] − KL(q(Z|A,X) ∥ p(Z))
```

```python
class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1      = GCNConv(in_channels, hidden_channels)
        self.conv_mu    = GCNConv(hidden_channels, out_channels)
        self.conv_logstd = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index).relu()
        return self.conv_mu(h, edge_index), self.conv_logstd(h, edge_index)

class VGAE(torch.nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def reparameterize(self, mu, logstd):
        if self.training:
            return mu + torch.randn_like(mu) * torch.exp(logstd)
        return mu

    def encode(self, x, edge_index):
        self.mu, self.logstd = self.encoder(x, edge_index)
        self.logstd = self.logstd.clamp(max=10)
        return self.reparameterize(self.mu, self.logstd)

    def decode(self, z, edge_index):
        src, dst = edge_index
        return (z[src] * z[dst]).sum(dim=-1)

    def decode_all(self, z):
        return torch.sigmoid(z @ z.t())

    def kl_loss(self):
        return -0.5 * torch.mean(
            torch.sum(1 + 2*self.logstd - self.mu.pow(2) - self.logstd.exp().pow(2), dim=1)
        )

    def recon_loss(self, z, pos_edge_index, neg_edge_index=None):
        pos_loss = -torch.log(torch.sigmoid(self.decode(z, pos_edge_index)) + 1e-15).mean()
        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, z.size(0), pos_edge_index.size(1))
        neg_loss = -torch.log(1 - torch.sigmoid(self.decode(z, neg_edge_index)) + 1e-15).mean()
        return pos_loss + neg_loss
```

**What breaks:**

- **O(n²) decoding:** the inner-product decoder evaluates all node pairs — infeasible for large graphs.
- **No validity guarantee:** nothing prevents the decoder from producing chemically invalid graphs.
- **No global graph structure:** the latent space has one vector per node; there is no graph-level variable that captures global properties (molecular weight, ring count, connectivity).

---

## 7. Sequential Graph Generation

### GraphRNN (You et al., 2018)

**The problem:** how do you generate a graph node-by-node while respecting the permutation ambiguity? Any ordering is arbitrary, but you need *some* ordering to generate sequentially.

**The core insight:** use a canonical BFS ordering. In a BFS-ordered adjacency matrix, a new node can only connect to nodes within a limited bandwidth — nodes encountered recently in the BFS traversal. This turns an O(n²) connection sequence into O(n·M) where M is the BFS bandwidth (small for sparse graphs).

**The mechanics:**
```
For each new node v_i:
  1. Graph-level RNN updates hidden state: h_i = f_graph(h_{i-1})
  2. Edge-level RNN generates binary sequence: e_{i,1}, e_{i,2}, ..., e_{i,i-1}
     where e_{i,j} = 1 means edge (v_i, v_j) exists
  3. Stop when EOS token generated
```

**What breaks:** sequential generation is slow at inference — generating n nodes requires n² RNN steps in the worst case. BFS bandwidth limits this but makes rare long-range connections impossible.

### GRAN (Graph Recurrent Attention Network)

**The problem:** GraphRNN generates one node at a time, treating the generation as a sequence of independent binary decisions. For complex graphs, the decision of whether node v_i connects to v_j depends on the global topology, not just the local history captured by the RNN.

**The core insight:** generate a *block* of nodes at each step and use graph attention over the partially-generated graph to score edges within the block. The attention captures global topology; block generation enables parallelism.

**The mechanics:**
```
At step t:
  - Candidate block B_t = {v_{k+1}, ..., v_{k+b}}
  - Run GAT over current graph + candidates → score edges
  - Sample edges between B_t and existing nodes
  - Add B_t to graph, repeat
```

**What breaks:** the graph attention at each generation step over the full partial graph is O(n²) in the number of existing nodes — expensive for large graphs. Block size b must be chosen carefully: too large → complex edge dependencies; too small → sequential bottleneck.

---

## 8. Diffusion-Based Graph Generation

### GDSS (Graph Diffusion Score-Based Model, 2022)

**The problem:** diffusion models produce high-quality continuous data (images, audio). Can you apply them to graphs, where both node features and the adjacency structure need to be generated jointly?

**The core insight:** diffuse both the node feature matrix X and the adjacency matrix A simultaneously. The score network must be permutation-equivariant to handle the n! orderings. Use a GNN on the current noisy (X_t, A_t) to compute scores.

**The mechanics:**
```
Forward:  (X_T, A_T) ~ N(0, I) × N(0, I)
Reverse:  score model s_θ learns ∇_{X,A} log p_t(X, A)
          jointly denoises both X and A at each reverse step
```

**What breaks:** the score function must be permutation-equivariant — standard architectures that condition on a fixed node ordering are not valid. Enforcing equivariance adds architectural constraints and limits the expressivity of the score network.

### DiGress (2023)

**The problem:** Gaussian diffusion on adjacency matrices produces real-valued, dense "noisy" adjacency matrices mid-diffusion. Real graphs are binary and sparse. Working in continuous space means the model must learn to produce integer-valued, sparse graphs as a special case of continuous generation.

**The core insight:** operate in *discrete space*. Define a categorical Markov noise process that randomly adds/removes edges and changes node types. The forward process gradually randomizes the graph toward a uniform distribution over graphs. The reverse process is a denoising transformer that predicts the clean graph directly.

**The mechanics:**
```
Forward:  corrupt by randomly adding/removing edges with probability β_t
Reverse:  transformer predicts G_0 from G_t directly (denoising approach, not score-based)
```

DiGress naturally handles discrete node/edge attributes and produces sparser, more valid graphs than continuous diffusion.

**What breaks:** the forward process (uniform edge noise) doesn't reflect any meaningful "less noisy" intermediate state — unlike image diffusion where partially-denoised images are recognizable. The model must learn to decode directly from arbitrarily corrupted graphs, making the learning problem harder.

---

## 9. Molecule Generation

### Representations

| Format | Example | Pros | Cons |
|---|---|---|---|
| SMILES string | `CC(=O)Oc1ccccc1C(=O)O` | Compact, standard | Invalid strings common, not unique |
| Graph | Nodes=atoms, Edges=bonds | Natural, validity-checkable | Permutation invariance |
| 3D point cloud | Atom xyz coordinates | Captures geometry | Loses bonding topology |

### Validity Constraints

A generated molecular graph is chemically valid if:
- Each atom satisfies its valency (Carbon: degree ≤ 4, Nitrogen ≤ 3, etc.)
- The graph is connected
- No invalid bond types

These constraints are hard to enforce during generation. Common approaches: post-hoc validity filtering, valency-aware decoding (mask invalid bond actions), or junction tree decomposition.

### Junction Tree VAE (JT-VAE, Jin et al. 2018)

**The problem:** atom-by-atom generation produces many chemically invalid molecules — the model must learn valency rules implicitly. Near-100% validity is hard to achieve.

**The core insight:** decompose molecules into chemically valid building blocks (ring systems and functional groups from a fixed vocabulary). Generate the *junction tree* of these building blocks first, then assemble them into a full molecule by resolving attachment points. Because each motif is valid by construction, validity is nearly guaranteed.

**The mechanics:**
1. Build junction tree: generate which motifs and how they connect (tree structure).
2. Assemble: resolve how motifs attach to each other (bond connection).

Near-100% validity on standard benchmarks.

**What breaks:** the vocabulary of motifs must be pre-computed from the training corpus. Molecules containing ring systems or functional groups not seen during training cannot be generated — novelty is bounded by the vocabulary.

---

## 10. Evaluation Metrics

### Graph Generation (General)

| Metric | Description |
|---|---|
| **Validity** | Fraction satisfying domain constraints (e.g., connectivity) |
| **Uniqueness** | Fraction of valid generated graphs that are non-duplicate |
| **Novelty** | Fraction not present in the training set |
| **Degree distribution** | MMD between degree histograms of generated vs. real graphs |
| **Clustering coefficient** | MMD on local clustering distributions |
| **Orbit statistics** | MMD on counts of 4-node subgraph patterns |

MMD (Maximum Mean Discrepancy) is the standard distance for comparing graph statistic distributions.

### Molecule Generation

| Metric | Description |
|---|---|
| **Validity** | Fraction passing RDKit sanitization |
| **Uniqueness** | Fraction unique among valid samples |
| **Novelty** | Fraction not in training set |
| **FCD** | Fréchet ChemNet Distance — distribution-level quality metric; lower is better |
| **SA Score** | Synthetic accessibility (1=easy, 10=hard) |
| **QED** | Quantitative estimate of drug-likeness (0–1) |

FCD is the molecule generation analogue of FID for images: it computes the Fréchet distance between the distributions of generated and real molecules in a learned ChemNet embedding space. A model with high validity/uniqueness but wrong chemical distribution shows high FCD.

**What breaks with validity-only metrics:** a model that always generates methane (CH₄) achieves 100% validity, 0% uniqueness, and 0% novelty — useless. Always report all three: validity, uniqueness, novelty. Add FCD and QED for drug discovery contexts.

---

## 11. Key Interview Points

**DTDG vs CTDG — when does each apply?**
DTDG: data is naturally collected at regular intervals (traffic sensors, daily social snapshots) — discretization is free. CTDG: events happen at irregular times where the timestamp itself is informative (financial transactions, social media posts). CTDG is more expressive but harder to implement because you need time encodings and temporal neighborhood lookups.

**EvolveGCN's key idea**
EvolveGCN evolves the GCN *weight matrices* with an RNN — the weight matrix IS the hidden state. This decouples temporal modeling from node identity. New nodes can appear at any snapshot without needing an initialization scheme; the weight matrix evolves based on the structural properties of the graph, not on which specific nodes are present.

**TGN memory module — why is it necessary?**
Without memory, a node's embedding at query time is computed only from its current temporal neighborhood — a few recent interactions. The memory vector summarizes the node's *entire* interaction history in a fixed-size vector. Nodes that interacted last week still carry that information forward. Without memory, a rarely-active node appears as a new node at every query.

**TGAT time encoding — what does it actually do?**
It maps a scalar time difference Δt to a vector in ℝ^d using learnable Fourier features: `[cos(wΔt + b), sin(wΔt + b)]`. The key property: the kernel k(Δt1, Δt2) = φ(Δt1)·φ(Δt2) is a function only of |Δt1 − Δt2|, so the representation captures *how much time has passed* without depending on absolute timestamps. Using relative time `t − t'` instead of absolute timestamps makes the model generalizable across different time ranges.

**Temporal link prediction — why must you use time-based splits?**
Random splits cause leakage: events from the future appear in training data. The model sees what connections will exist at test time, inflating metrics. In real deployment, the model only ever has access to past events. Time-based splits simulate this condition.

**Permutation invariance in graph generation**
The same graph has n! valid adjacency matrix representations — one per node ordering. A generative model that conditions on a fixed ordering (e.g., the one in the training set) will fail to generate the same graph under a different ordering. GraphRNN addresses this with canonical BFS ordering. GDSS and DiGress use permutation-equivariant architectures (GNNs) for the score/denoising network, so the output is the same graph regardless of the input ordering.

**VGAE decoding: why does inner product predict edges?**
The VGAE latent space is regularized by a Gaussian prior; similar nodes have close latent vectors. The inner product z_iᵀ z_j = ‖z_i‖ ‖z_j‖ cos(θ) is large when vectors are close (small angle). Through sigmoid, this becomes the edge probability. The geometry of the learned latent space directly encodes graph topology.

**Validity alone is insufficient for molecule generation**
A model that generates only methane achieves 100% validity. Validity, uniqueness, and novelty together characterize a useful model. FCD additionally measures whether the distribution of generated molecules matches the training distribution — a model with high V/U/N but wrong chemical diversity will show high FCD. Distinguishing evaluation metrics from training objectives is critical: models optimize ELBO or score matching, not validity directly.
