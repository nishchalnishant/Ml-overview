# Graph Neural Networks

---

## Table of Contents

1. [Why Graph Neural Networks?](#1-why-graph-neural-networks)
2. [Graph Basics](#2-graph-basics)
3. [Challenges with Graphs](#3-challenges-with-graphs)
4. [Graph Convolutional Networks (GCN)](#4-graph-convolutional-networks-gcn)
5. [GraphSAGE — Inductive Learning](#5-graphsage--inductive-learning)
6. [Graph Attention Networks (GAT)](#6-graph-attention-networks-gat)
7. [Message Passing Neural Networks (MPNN)](#7-message-passing-neural-networks-mpnn)
8. [Spectral vs Spatial Graph Convolutions](#8-spectral-vs-spatial-graph-convolutions)
9. [Graph Pooling](#9-graph-pooling)
10. [Applications](#10-applications)
11. [Knowledge Graph Embeddings](#11-knowledge-graph-embeddings)
12. [GNNs in LLMs](#12-gnns-in-llms)
13. [Scalability Challenges](#13-scalability-challenges)
14. [Common Interview Questions](#14-common-interview-questions)

---

## 1. Why Graph Neural Networks?

Most of the data you interact with in ML is either a fixed-size vector (tabular), a grid (images), or a sequence (text, time series). CNNs exploit the grid structure of pixels. Transformers exploit the sequence structure of tokens. But a lot of the world's most interesting data is neither — it's relational. It's a graph.

**Social networks.** You're not just a feature vector. You're a person with friends, who have friends, who belong to communities. The signal for "will this user like this content?" lives not just in your profile, but in the structure around you. Who follows whom? Which clusters do you belong to? A vanilla feedforward network that ignores that structure throws away half the signal.

**Molecules.** A drug molecule is atoms (nodes) connected by chemical bonds (edges). The property you care about — toxicity, solubility, binding affinity — emerges from both atom identity and bonding topology. You can't flatten a molecule into a fixed-size vector without losing structural information. Two molecules can have the same atoms but different connectivity and behave completely differently (think structural isomers).

**Knowledge graphs.** Systems like Freebase, Wikidata, or the Google Knowledge Graph represent facts as triples: (Barack Obama, born_in, Hawaii). Here entities are nodes, relationships are typed edges. Reasoning over these graphs — "who are all US presidents born in the same state as Obama?" — requires traversing and aggregating across structured relational data.

**The common thread:** in all these cases, the input has variable size, irregular structure, and the important features are defined by relationships, not just attributes. GNNs are the family of architectures built to handle exactly this.

> **Core idea:** Learn node representations by iteratively aggregating information from a node's local neighborhood. After k layers, each node's embedding encodes information from its k-hop neighborhood.

---

## 2. Graph Basics

Before diving into architectures, lock down the vocabulary. Interviewers will test this.

### Formal Definition

A graph is G = (V, E) where:
- **V** = set of nodes (vertices), |V| = N
- **E** ⊆ V × V = set of edges

Each node v ∈ V may have a feature vector **x**_v ∈ R^d. Each edge (u, v) ∈ E may have a feature vector **e**_{uv}.

### Adjacency Matrix

The adjacency matrix **A** ∈ {0,1}^{N×N} where:

```
A[i][j] = 1  if edge (i, j) exists
A[i][j] = 0  otherwise
```

For weighted graphs, A[i][j] holds the edge weight. This matrix is the primary way graphs are represented in code, though you'll also see edge lists and sparse formats.

**Key property:** For an undirected graph, **A** is symmetric: A[i][j] = A[j][i]. For a directed graph, it's generally not.

### Degree

The **degree** of a node is the number of edges connected to it.

- **Undirected:** deg(v) = Σ_j A[v][j]  
- **Directed:** in-degree = Σ_j A[j][v], out-degree = Σ_j A[v][j]

The **degree matrix** D is a diagonal matrix where D[i][i] = deg(i). It shows up constantly in GCN math.

### Directed vs Undirected

- **Undirected:** friendships, molecular bonds, co-citation networks. Symmetry is baked in.
- **Directed:** Twitter follows, citation direction ("paper A cites paper B"), dependency graphs. Direction carries semantic meaning.

### Other Structural Concepts

| Concept | Definition | Why it matters |
|---|---|---|
| Neighborhood N(v) | All nodes directly connected to v | Defines the scope of one GNN layer |
| Path | Sequence of edges connecting two nodes | Determines information flow depth |
| Connected component | Maximal subgraph where all nodes are reachable | Many real graphs have multiple components |
| Diameter | Longest shortest path in the graph | Bounds how many layers you need |
| Clustering coefficient | Fraction of a node's neighbors that are also connected to each other | Measures local "cliqueness" |

### The Laplacian

The **graph Laplacian** L = D - A is central to spectral methods (covered in Section 8).

The **normalized Laplacian** is:

```
L_norm = D^{-1/2} L D^{-1/2} = I - D^{-1/2} A D^{-1/2}
```

Its eigenvalues lie in [0, 2] and encode the graph's frequency structure — analogous to Fourier frequencies for regular grids.

---

## 3. Challenges with Graphs

Why can't you just use a CNN or Transformer on graphs? Three fundamental problems:

### Variable Size

A graph can have 5 nodes or 5 million nodes. Molecular graphs for small drug-like molecules typically have 20-50 nodes; social graphs have billions. Unlike images (resized to 224×224) or text (padded to 512 tokens), there's no canonical "resize" operation for graphs that preserves structural meaning.

### No Natural Ordering

If I relabel the nodes of a graph — swap node 3 and node 7 — nothing about the graph has fundamentally changed. But if I do that to a feature matrix, the matrix changes. Any model you build needs to be **permutation invariant** (the output doesn't change if you relabel nodes) or **permutation equivariant** (the output transforms consistently with the relabeling).

Formally, for a graph-level prediction task:
```
f(PX, PAP^T) = f(X, A)   for any permutation matrix P
```

This rules out naive MLP or RNN approaches that depend on a fixed ordering.

### Node Feature Aggregation

For each node, you want to incorporate information from its neighbors. But different nodes have different numbers of neighbors — a celebrity on Twitter might have 50 million followers, an ordinary user might have 200. Your aggregation function needs to handle variable-size neighbor sets and produce a fixed-size output. Common solutions: mean, sum, max pooling over neighbors.

### Sparsity and Scale

Real graphs are massively sparse — a social network with 1 billion users might average 200 connections per user. The full adjacency matrix would be 10^18 entries, but only 2×10^11 are non-zero. Algorithms that naively materialize the full A matrix don't scale. GNN implementations lean heavily on sparse matrix operations.

---

## 4. Graph Convolutional Networks (GCN)

Kipf & Welling (2017) is the paper that put GNNs on the map for the broader ML community. The analogy: if CNNs convolve a filter over local pixel neighborhoods, GCNs convolve over local graph neighborhoods.

### The Message Passing Framework

Message passing is the unifying abstraction for almost all GNN architectures. Each layer performs two steps:

**Step 1 — Message:** For every edge (u, v), compute a message from u to v:

```
m_{u→v}^{(k)} = MSG^{(k)}(h_u^{(k-1)}, h_v^{(k-1)}, e_{uv})
```

**Step 2 — Aggregate + Update:** For each node v, aggregate the incoming messages and update its embedding:

```
h_v^{(k)} = UPDATE^{(k)}(h_v^{(k-1)}, AGG({m_{u→v}^{(k)} : u ∈ N(v)}))
```

After k layers, h_v^{(k)} encodes information from the k-hop neighborhood of v.

### GCN Formulation

The GCN layer (Kipf & Welling) is:

```
H^{(k+1)} = σ( Ã H^{(k)} W^{(k)} )
```

Where:
- H^{(k)} ∈ R^{N×d_k} = node feature matrix at layer k
- W^{(k)} ∈ R^{d_k × d_{k+1}} = learnable weight matrix
- σ = non-linearity (ReLU)
- Ã = D̃^{-1/2} Ã_raw D̃^{-1/2} = normalized adjacency with self-loops

The "with self-loops" part: Ã_raw = A + I. Adding I ensures each node aggregates its own features alongside its neighbors'. Without this, a node's previous-layer representation doesn't feed into its next-layer representation.

**Why the D̃^{-1/2} normalization?** Without it, nodes with high degree get larger aggregated signals, and the outputs explode for hubs. The symmetric normalization keeps magnitudes stable across nodes with different degrees.

### Aggregation Variants

Different aggregation functions have different properties:

**Mean aggregation:**
```
h_v^{(k)} = σ( W · MEAN({h_u^{(k-1)} : u ∈ N(v) ∪ {v}}) )
```
- Ignores neighborhood size
- Good when you want degree-invariant representations
- Fails to distinguish a node with 2 neighbors (A, B) from one with 100 neighbors that happen to have the same mean

**Sum aggregation:**
```
h_v^{(k)} = σ( W · SUM({h_u^{(k-1)} : u ∈ N(v)}) )
```
- Sensitive to degree
- Can distinguish neighborhood sizes — if all neighbors have distinct features, sum is injective
- GIN (Graph Isomorphism Network) proves sum is strictly more expressive than mean

**Max aggregation:**
```
h_v^{(k)} = σ( W · MAX({h_u^{(k-1)} : u ∈ N(v)}) )
```
- Captures the "most extreme" feature value across neighbors
- Good for detecting the presence of specific structural features
- Loses information about multiplicity

### Layer-by-Layer Neighborhood Expansion

This is the key intuition for depth in GNNs. After:
- **1 layer:** each node sees its direct neighbors (1-hop)
- **2 layers:** each node sees neighbors of neighbors (2-hop)
- **k layers:** each node sees its entire k-hop neighborhood

For a molecule, 2-3 layers is often enough because most relevant chemical context is within 3 bonds. For social networks, this is where things get tricky — the 6-degrees-of-separation phenomenon means 6 layers would technically connect most nodes, but the 6-hop neighborhood can be the entire graph.

### PyTorch Example (GCN Layer from Scratch)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x, adj_norm):
        # adj_norm: pre-computed D^{-1/2} (A + I) D^{-1/2}, sparse or dense
        # x: (N, in_features)
        support = self.linear(x)              # (N, out_features)
        out = torch.sparse.mm(adj_norm, support)  # (N, out_features)
        return F.relu(out)

class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.5):
        super().__init__()
        self.gc1 = GCNLayer(in_dim, hidden_dim)
        self.gc2 = GCNLayer(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, x, adj_norm):
        x = self.gc1(x, adj_norm)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj_norm)
        return F.log_softmax(x, dim=1)  # for node classification
```

### Pre-computing the Normalized Adjacency

```python
import scipy.sparse as sp
import numpy as np

def normalize_adj(adj):
    """Symmetric normalization: D^{-1/2} (A + I) D^{-1/2}"""
    adj = adj + sp.eye(adj.shape[0])          # add self-loops
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv_sqrt = np.power(rowsum, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat = sp.diags(d_inv_sqrt)
    return d_mat @ adj @ d_mat                # sparse matrix
```

### Limitations of Basic GCN

1. **Transductive only:** The normalized adjacency is computed for the full training graph. If a new node arrives at inference time, you need to re-compute the full normalization. This is the motivation for GraphSAGE.
2. **Over-smoothing:** Stack too many layers and all node representations converge to the same value (analogous to a random walk mixing). Empirically, 2-3 layers is often optimal.
3. **Fixed equal weights for all neighbors:** A node's close friends and casual acquaintances get the same weight. GAT fixes this.

---

## 5. GraphSAGE — Inductive Learning

Hamilton et al. (2017). The name: SAmple and aggreGatE.

### The Core Insight

Standard GCN computes embeddings using the full graph adjacency. This is **transductive** — you can only generate embeddings for nodes that were present during training. This is a problem for:

- **Evolving graphs:** New users join a social network every day.
- **Large graphs:** You can't fit the full graph in memory.
- **Generalization:** You want to apply the same learned function to unseen graph structures.

GraphSAGE makes GNN learning **inductive** by learning a *function* (a set of aggregator parameters) that can be applied to any neighborhood, regardless of whether those nodes were seen during training.

### The Algorithm

**Training phase** (per node v, per mini-batch):

```
For each layer k = 1 ... K:
    1. Sample a fixed-size neighborhood S_v ⊆ N(v)  (e.g., sample 25 neighbors)
    2. Compute neighbor aggregate:
         h_N(v)^k = AGG_k({h_u^{k-1} : u ∈ S_v})
    3. Concatenate and transform:
         h_v^k = σ( W^k · CONCAT(h_v^{k-1}, h_N(v)^k) )
    4. L2-normalize:
         h_v^k = h_v^k / ||h_v^k||_2
```

**Inference on a new node:** Given a new node v with features x_v and its observed neighbors, run the same aggregator function. No re-training, no recomputing global normalization.

### Why Neighborhood Sampling?

Without sampling, the computational graph for a single node at depth k grows exponentially: |N(v)|^k nodes. For a social graph with average degree 200 and k=2, that's 40,000 nodes per training example. With k=3, it's 8 million. Sampling fixes the receptive field to a manageable constant — typically 25 at depth 1, 10 at depth 2.

### Aggregator Variants in GraphSAGE

**Mean aggregator:**
```python
h_v^k = σ( W · MEAN([h_v^{k-1}] ∪ {h_u^{k-1} : u ∈ S_v}) )
```
Roughly equivalent to the GCN update (no concat, just mean including self).

**LSTM aggregator:**
```python
# Shuffle neighbors randomly, feed through LSTM, take final hidden state
h_N(v) = LSTM([h_u^{k-1} for u in shuffle(S_v)])
```
LSTMs are not permutation-invariant by design, so a random shuffle approximates it. Not ideal theoretically but works well in practice.

**Pooling aggregator:**
```python
# Transform each neighbor independently, then max-pool
h_N(v) = MAX({σ(W_pool · h_u^{k-1} + b) : u ∈ S_v})
```

### Training Objective

GraphSAGE can be trained:

- **Supervised:** cross-entropy for node classification, standard backprop
- **Unsupervised:** graph-based loss that makes nearby nodes have similar embeddings:

```
J(z_v) = -log(σ(z_v^T z_u)) - Q · E_{v_n ~ P_n}[log(σ(-z_v^T z_{v_n}))]
```

Where u is a neighbor of v (positive pair), v_n is a random negative sample, Q is the number of negative samples.

### PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

# For mini-batch training with neighbor sampling:
from torch_geometric.loader import NeighborLoader

loader = NeighborLoader(
    data,
    num_neighbors=[25, 10],   # sample 25 neighbors at hop 1, 10 at hop 2
    batch_size=1024,
    input_nodes=data.train_mask,
)
```

---

## 6. Graph Attention Networks (GAT)

Veličković et al. (2018). The insight: not all neighbors are equally informative. Your best friend's opinion should carry more weight than a spam account that followed you.

### Attention Over Neighbors

For each node v, GAT learns an **attention coefficient** α_{vu} for each neighbor u. The update becomes:

```
h_v^{(k)} = σ( Σ_{u ∈ N(v) ∪ {v}} α_{vu} · W h_u^{(k-1)} )
```

Where the attention weights sum to 1: Σ_u α_{vu} = 1.

### Computing Attention Coefficients

**Step 1:** Project node features:
```
z_v = W h_v
```

**Step 2:** Compute unnormalized attention between v and each neighbor u:
```
e_{vu} = LeakyReLU( a^T · CONCAT(z_v, z_u) )
```
Here **a** ∈ R^{2d'} is a learnable attention vector.

**Step 3:** Normalize with softmax over v's neighborhood:
```
α_{vu} = softmax_u(e_{vu}) = exp(e_{vu}) / Σ_{w ∈ N(v) ∪ {v}} exp(e_{vw})
```

**Step 4:** Weighted aggregation:
```
h_v' = σ( Σ_{u ∈ N(v) ∪ {v}} α_{vu} · W h_u )
```

### Multi-Head Attention

Just like in Transformers, multiple attention heads capture different relationship types:

```
h_v' = CONCAT_{k=1}^{K} σ( Σ_{u ∈ N(v)} α_{vu}^k · W^k h_u )
```

For the final layer, use averaging instead of concatenation:
```
h_v' = σ( (1/K) Σ_{k=1}^{K} Σ_{u ∈ N(v)} α_{vu}^k · W^k h_u )
```

### Connection to Transformer Attention

This is worth knowing for interviews. The standard Transformer attention:

```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
```

GAT attention is essentially doing the same thing on a graph:
- **Q** = the query node v (through W_Q)
- **K** = the key neighbor u (through W_K)  
- **V** = the value (W_V h_u)
- The **mask** is the graph adjacency — you only attend to actual neighbors

The difference: in a Transformer, every token attends to every other token (full attention). In GAT, attention is restricted to the graph neighborhood. This makes GAT much more efficient on graphs where each node has a small, structured neighborhood.

GAT v2 (Brody et al., 2022) fixes a subtle expressiveness problem in the original: in GATv1, the ranking of attention scores can't depend on the query node for the same set of keys (the attention is "static" in a sense). GATv2 fixes this by changing the order of operations.

### PyTorch Example

```python
from torch_geometric.nn import GATConv

class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 heads=8, dropout=0.6):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads,
                             dropout=dropout)
        self.conv2 = GATConv(hidden_channels * heads, out_channels,
                             heads=1, concat=False, dropout=dropout)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
```

---

## 7. Message Passing Neural Networks (MPNN)

Gilmer et al. (2017). This is the general unifying framework — most GNN architectures are special cases of it.

### The General Framework

An MPNN operates in two phases:

**Message Passing Phase** (T steps):

```
m_v^{t+1} = Σ_{w ∈ N(v)} M_t(h_v^t, h_w^t, e_{vw})
```

**Update phase:**

```
h_v^{t+1} = U_t(h_v^t, m_v^{t+1})
```

Here:
- M_t = message function (any differentiable function)
- U_t = update function (any differentiable function, often a GRU)
- e_{vw} = edge features between v and w

**Readout Phase** (for graph-level tasks):

```
ŷ = R({h_v^T : v ∈ G})
```

R is a readout function that produces a graph-level representation, e.g., sum over all final node embeddings.

### How Other Models Fit In

| Model | Message function M_t | Update function U_t |
|---|---|---|
| GCN | h_w (neighbor features) | Linear + ReLU |
| GraphSAGE | h_w | Linear(concat(h_v, mean(neighbors))) |
| GAT | α_{vw} · W h_w | Sum of weighted messages |
| GGNN | h_w (no edge features) | GRU |

### Edge Features Matter

One advantage of the MPNN framing: edge features are first-class citizens. In chemistry, bond type (single, double, aromatic) is critical. The message function can incorporate both the neighbor's state and the edge type:

```python
# Message incorporating edge features
def message(h_v, h_w, e_vw):
    return nn.Linear(h_w.size(-1) + e_vw.size(-1), hidden_dim)(
        torch.cat([h_w, e_vw], dim=-1)
    )
```

### Gated Graph Neural Network (GGNN)

Li et al. (2016) — uses GRU as the update function:

```
m_v^t = Σ_{w ∈ N(v)} W_e · h_w^t
h_v^{t+1} = GRU(h_v^t, m_v^t)
```

This allows information to flow over multiple steps while gating out irrelevant information, similar to sequence models.

---

## 8. Spectral vs Spatial Graph Convolutions

Two distinct philosophical approaches to defining convolution on graphs. You'll get asked about this.

### Spectral Approach

Motivated by signal processing: define convolution in the frequency domain via the graph Fourier transform.

The **eigenvector decomposition** of the Laplacian L = U Λ U^T gives:
- U = matrix of eigenvectors (Fourier basis)
- Λ = diagonal matrix of eigenvalues (frequencies)

The graph Fourier transform of a signal x:
```
x̂ = U^T x
```

Spectral convolution with a filter g:
```
x *_G g = U · (U^T x ⊙ U^T g) = U · g_θ(Λ) · U^T x
```

Where g_θ(Λ) is a diagonal filter matrix parameterized by θ.

**Problem:** O(N^2) computation for the full eigenvector decomposition. Not scalable.

**ChebNet** (Defferrard et al., 2016) approximates the filter with Chebyshev polynomials of degree K:

```
g_θ(L) ≈ Σ_{k=0}^{K} θ_k T_k(L̃)
```

Where L̃ = 2L/λ_max - I is the rescaled Laplacian. This makes computation O(K|E|) — linear in the number of edges.

**GCN** simplifies ChebNet further: take K=1 (first-order approximation), assume λ_max ≈ 2, and you recover the GCN formula. So GCN is a specific spectral method with a first-order Chebyshev approximation.

### Spatial Approach

Forget eigendecompositions. Just directly define how a node aggregates information from its spatial neighbors. This is what GraphSAGE, GAT, GIN, and most modern methods do.

**Pros of spatial:**
- Naturally inductive — parameters are shared across all nodes
- Handles graphs of arbitrary size without recomputing eigendecompositions
- More computationally efficient in practice
- Edge features are easy to incorporate

**Pros of spectral:**
- Theoretically grounded in signal processing
- Can learn global frequency patterns that spatial methods miss
- Principled way to think about "smoothness" of node representations

### Practical Guidance

In most industrial and research applications today, spatial methods dominate. The requirement to re-compute eigendecompositions for new graphs (or even new graph sizes) makes spectral methods impractical for inductive settings. Unless you're specifically working on signals on fixed graphs (e.g., brain connectivity, sensor networks with fixed topology), default to spatial GNNs.

---

## 9. Graph Pooling

A single graph often represents one data point (e.g., a molecule). You need a graph-level embedding, not just node-level embeddings. Graph pooling maps a set of node embeddings to a single vector.

### Global Pooling

The simplest approach: aggregate all node embeddings into one.

**Global mean pooling:**
```
h_G = (1/|V|) Σ_{v ∈ V} h_v
```
Invariant to graph size, ignores structural information about which nodes are more important.

**Global sum pooling:**
```
h_G = Σ_{v ∈ V} h_v
```
More expressive than mean — can distinguish graphs that have the same mean but different total "mass."

**Global max pooling:**
```
h_G = MAX_{v ∈ V}({h_v})  (element-wise max)
```
Captures the most prominent feature across all nodes.

**Global add + readout with learnable weights:**
```python
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool

# After computing node embeddings:
graph_embedding = global_mean_pool(node_embeddings, batch)  # batch: node-to-graph assignment
```

### Hierarchical Pooling

Global pooling discards all graph structure above the node level. Hierarchical pooling repeatedly coarsens the graph, analogous to pooling layers in a CNN.

**DiffPool** (Ying et al., 2018) — the canonical hierarchical pooling method.

The idea: at each pooling layer, learn a soft cluster assignment matrix S ∈ R^{N×C} that maps N nodes to C clusters.

```
S^(l) = softmax( GNN_pool^(l)(A^(l), X^(l)) )
```

Then the pooled node features and adjacency are:
```
X^(l+1) = S^(l)^T · Z^(l)     (Z^(l) = GNN_embed^(l)(A^(l), X^(l)))
A^(l+1) = S^(l)^T · A^(l) · S^(l)
```

This creates a coarser graph with C nodes, where each "super-node" aggregates a cluster of the original nodes.

**Training the pooling:** DiffPool adds two auxiliary losses:
1. **Link prediction loss:** encourages adjacent nodes to be in the same cluster
2. **Entropy regularization:** encourages sharp (near-binary) assignments rather than fuzzy ones

**Limitations of DiffPool:** O(N^2) memory for the assignment matrix. For large graphs, this is prohibitive.

### Alternatives to DiffPool

- **MinCutPool:** Uses spectral clustering loss for cluster assignments
- **Top-K Pooling:** Select the top-k most important nodes based on a learned scalar score, drop the rest
- **SAGPooling:** Uses graph attention scores to rank and select nodes

```python
from torch_geometric.nn import TopKPooling

class HierarchicalGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.pool1 = TopKPooling(hidden_channels, ratio=0.8)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.pool2 = TopKPooling(hidden_channels, ratio=0.8)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        return global_mean_pool(x, batch)
```

---

## 10. Applications

### Molecular Property Prediction (Drug Discovery)

This is arguably the application that generated the most GNN research in the late 2010s.

**The problem:** Given a drug candidate molecule, predict its properties: solubility, toxicity, binding affinity to a target protein, ADMET properties (Absorption, Distribution, Metabolism, Excretion, Toxicity).

**Why GNNs?** Molecules are graphs. Atoms = nodes (features: atomic number, charge, hybridization), bonds = edges (features: bond type, aromatic). The key insight from chemistry: molecular properties are **local** — a functional group's contribution to solubility depends on the nearby chemical environment, not the entire molecule. This maps naturally to the local neighborhood aggregation of GNNs.

**Benchmark:** QM9 (quantum mechanical properties of 134k small molecules), MoleculeNet (broad benchmark suite). Schütt et al.'s SchNet and DimeNet pushed the state of the art here.

**AlphaFold connection:** While AlphaFold 2 uses attention on residue sequences + pairwise features, not a pure GNN, the subsequent AlphaFold 3 incorporates graph-based reasoning for molecular structure prediction. GNNs are the backbone of many docking and binding prediction models.

```python
import torch
from torch_geometric.nn import NNConv, global_add_pool
from torch_geometric.data import Data

# Molecule as a PyTorch Geometric graph
# node_features: (num_atoms, atom_feature_dim)
# edge_index: (2, num_bonds)
# edge_features: (num_bonds, bond_feature_dim)

class MoleculeGNN(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, out_dim):
        super().__init__()
        # NNConv: edge-conditioned convolution (message depends on edge features)
        edge_nn1 = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim * hidden_dim),
        )
        self.conv1 = NNConv(node_dim, hidden_dim, edge_nn1)
        edge_nn2 = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim * hidden_dim),
        )
        self.conv2 = NNConv(hidden_dim, hidden_dim, edge_nn2)
        self.readout = nn.Linear(hidden_dim, out_dim)

    def forward(self, data):
        x, edge_index, edge_attr, batch = (
            data.x, data.edge_index, data.edge_attr, data.batch
        )
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x = global_add_pool(x, batch)   # graph-level embedding
        return self.readout(x)
```

### Recommendation Systems — Pinterest PinSage

Hamilton et al. (2018), deployed at Pinterest with 3B nodes and 18B edges.

**The problem:** Pinterest is a graph of users, pins, and boards. A user saves a pin to a board; that's an edge. Given a pin, recommend visually and semantically similar pins.

**Why not just use image embeddings?** Two pins of coffee mugs might look identical but have completely different contexts — one is on a "home brewing" board followed by coffee nerds, the other is on a "party supplies" board. Graph context encodes usage patterns that raw image features miss.

**PinSage contributions:**
1. **Random walk-based neighbor sampling:** Instead of uniformly sampling neighbors, use biased random walks — nodes visited more frequently are more "important" neighbors. This approximates personalized PageRank.
2. **Importance-based neighborhood:** Assign importance weights to sampled neighbors proportional to visit frequency. Weighted aggregation follows.
3. **Curriculum training:** Start with easy negatives (random items), progressively use harder negatives (items that are similar but not relevant).

**At scale:** Offline training generates embeddings for all pins. Online serving just does nearest-neighbor lookup in the embedding space — no graph traversal needed at inference time.

### Knowledge Graphs & Entity Disambiguation

Knowledge graphs (KGs) like Wikidata represent facts as triples (subject, relation, object). Example: (Paris, capital_of, France).

**Entity disambiguation (entity linking):** Given a mention "Washington" in text, determine whether it refers to the city, the state, George Washington, or Denzel Washington. The KG around each entity candidate provides rich structural context.

**GNN approach:** Build a local subgraph around each entity candidate (its neighbors in the KG, their types, their connections). Run a GNN to get a context-aware embedding for each candidate. The candidate whose embedding is most compatible with the textual context wins.

**Relation prediction / link prediction:** Given an incomplete KG, predict missing links: (Paris, ?, France) → capital_of. This is covered in Section 11 with TransE and RotatE.

### Social Network Analysis

**Fraud detection (Graph fraud detection):** Financial transaction graphs where nodes are accounts and edges are transactions. Fraudsters often form tightly-connected communities (because they control the accounts on both sides of synthetic transactions). GNNs over the transaction graph propagate fraud signals: if your neighbors are flagged as fraudulent, your risk score increases. This is hard to game because you'd need to also control a large connected neighborhood.

**Community detection:** Learn node embeddings such that nodes in the same community are close in embedding space. SAGE or GCN with community labels, or unsupervised contrastive learning on graph structures.

**Influence propagation:** Given a set of "seed" nodes for a campaign, which additional nodes will be activated through social influence? GNNs can model the diffusion dynamics by learning from historical cascade data.

### Fraud Detection in Detail

```python
# Binary node classification: fraud vs. legitimate
# Graph: financial transaction network
# Node features: account age, transaction velocity, device fingerprint features
# Edge features: transaction amount, timestamp, channel (web/mobile/ATM)

class FraudGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels)
        self.classifier = nn.Linear(hidden_channels, 2)

    def forward(self, x, edge_index):
        # 3 layers = look 3 hops away in transaction network
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        return self.classifier(x)
```

Key challenge: **class imbalance**. In fraud detection, fraudulent transactions might be 0.1% of all transactions. Strategies: weighted cross-entropy, oversampling fraud nodes, focal loss.

---

## 11. Knowledge Graph Embeddings

These are embedding methods specifically for knowledge graph completion — predicting missing triples (h, r, t) where h = head entity, r = relation, t = tail entity.

### The Setup

You have:
- A set of entities E
- A set of relations R
- A set of observed triples T ⊆ E × R × E

Goal: learn embeddings **h**, **r**, **t** ∈ R^d such that valid triples score high and invalid triples score low.

### TransE

Bordes et al. (2013) — the foundational model.

**Intuition:** Model a relation as a translation in embedding space. If (h, r, t) is a valid triple, then:

```
h + r ≈ t
```

**Score function:**
```
f(h, r, t) = -||h + r - t||   (negative L1 or L2 distance)
```

**Training:** Minimize margin-based loss:
```
L = Σ_{(h,r,t)∈T} Σ_{(h',r,t')∉T} max(0, γ + f(h,r,t) - f(h',r,t'))
```

Where γ is a margin, and (h', r, t') are negative samples (corrupted triples).

**Limitations of TransE:**
- Can't handle 1-to-many relations: if entity A has many cities (Paris, London, Berlin) via `capital_of_country`, TransE must push t_Paris ≈ t_London ≈ t_Berlin, which forces them to similar positions
- Symmetric relations: if (A, sibling_of, B) then (B, sibling_of, A). TransE would need h + r = t AND t + r = h, which forces r = 0

### RotatE

Sun et al. (2019) — models relations as rotations in complex space.

**Intuition:** Each entity is a complex vector, each relation is a phase rotation. Valid triples satisfy:

```
t = h ∘ r   (element-wise complex multiplication / rotation)
```

Where |r_i| = 1 (unit modulus constraint, so each dimension rotates, doesn't scale).

**Score function:**
```
f(h, r, t) = -||h ∘ r - t||
```

**Why RotatE is more expressive:**

| Relation pattern | TransE | RotatE |
|---|---|---|
| Symmetry (A↔B) | Cannot model | r_i = ±1 (rotation by 0 or π) |
| Antisymmetry (A→B but ¬B→A) | Yes (r ≠ 0) | Yes |
| Inversion (r₁ = r₂⁻¹) | Cannot model | r₁ ∘ r₂ = 1 |
| Composition (r₁ ∘ r₂ = r₃) | Partially | Yes |

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class TransE(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim, margin=1.0):
        super().__init__()
        self.entity_emb = nn.Embedding(num_entities, embedding_dim)
        self.relation_emb = nn.Embedding(num_relations, embedding_dim)
        self.margin = margin
        nn.init.uniform_(self.entity_emb.weight, -6/embedding_dim**0.5, 6/embedding_dim**0.5)
        nn.init.uniform_(self.relation_emb.weight, -6/embedding_dim**0.5, 6/embedding_dim**0.5)

    def score(self, h, r, t):
        h = F.normalize(self.entity_emb(h), p=2, dim=-1)
        r = self.relation_emb(r)
        t = F.normalize(self.entity_emb(t), p=2, dim=-1)
        return -torch.norm(h + r - t, p=1, dim=-1)

    def forward(self, pos_triples, neg_triples):
        pos_score = self.score(*pos_triples)
        neg_score = self.score(*neg_triples)
        loss = F.relu(self.margin + neg_score - pos_score).mean()
        return loss
```

### Other Notable KG Embedding Methods

- **DistMult:** Score = h^T diag(r) t (bilinear, symmetric)
- **ComplEx:** Complex-valued DistMult; handles asymmetric relations
- **RGCN (Relational GCN):** Uses separate weight matrices per relation type in the GCN update — bridges the gap between KG embeddings and GNNs
- **CompGCN:** Jointly embeds entities and relations using composition operations

---

## 12. GNNs in LLMs

This intersection is an active research area as of 2024-2025. Three main connections:

### Graph-Structured Reasoning

LLMs are good at processing text but struggle with multi-hop relational reasoning that requires consistent state tracking across many steps. "Who is the maternal grandfather of the US president who succeeded Nixon?" requires chaining (Nixon → Ford → Ford's mother → Ford's maternal grandfather), which is a path traversal on a knowledge graph.

Approaches:
1. **Retrieve + reason:** Extract a relevant subgraph from a KG, serialize it as text (triples or natural language), feed to LLM
2. **GNN + LLM hybrid:** Use a GNN to embed a retrieved subgraph, inject those embeddings as soft prompts or cross-attention context into the LLM
3. **Graph-of-Thought:** Structure the LLM's reasoning process itself as a graph — nodes are intermediate reasoning steps, edges encode dependencies

### Retrieval-Augmented Generation with Graphs (GraphRAG)

Microsoft's GraphRAG (2024) is the most prominent example:
1. Index a document corpus by building a knowledge graph from extracted entities and relationships
2. At query time, retrieve not just relevant documents but relevant graph communities (clusters of related entities)
3. Summarize each community's subgraph, pass summaries to LLM

Why this beats vanilla RAG for complex questions: "What are the main themes in this corpus?" requires understanding global structure, not just finding locally relevant passages. Community-based graph retrieval surfaces global patterns that chunk-based retrieval misses.

### LLMs as GNN Components

**LLM-as-node-feature-extractor:** Use an LLM to generate rich text-based features for each node in a graph (e.g., product descriptions → embeddings), then run a GNN on top. This is "text-attributed graphs" — combining the language understanding of LLMs with the structural reasoning of GNNs.

**LLM-as-graph-reasoner:** Frame graph tasks (node classification, link prediction) as text generation problems. Represent the graph as text, prompt the LLM. Works surprisingly well for small graphs; doesn't scale to large ones.

**G-Retriever (2024):** Given a question over a knowledge graph, retrieves a prize-collecting Steiner tree (the minimal subgraph connecting all relevant entities), encodes it with a GNN, then uses the GNN embedding as soft prompt tokens for an LLM.

### Practical Code: Text-Attributed Node Classification

```python
from transformers import AutoTokenizer, AutoModel
from torch_geometric.nn import GCNConv

class TextGNN(nn.Module):
    """LM for text features + GNN for graph structure."""
    def __init__(self, lm_model_name, hidden_dim, num_classes):
        super().__init__()
        self.lm = AutoModel.from_pretrained(lm_model_name)
        lm_dim = self.lm.config.hidden_size
        self.projection = nn.Linear(lm_dim, hidden_dim)
        self.gnn1 = GCNConv(hidden_dim, hidden_dim)
        self.gnn2 = GCNConv(hidden_dim, num_classes)

    def forward(self, input_ids, attention_mask, edge_index):
        # LM encodes text features for each node
        lm_out = self.lm(input_ids=input_ids, attention_mask=attention_mask)
        x = lm_out.last_hidden_state[:, 0, :]  # CLS token
        x = self.projection(x)
        # GNN propagates over graph structure
        x = F.relu(self.gnn1(x, edge_index))
        x = self.gnn2(x, edge_index)
        return x
```

---

## 13. Scalability Challenges

Scaling GNNs to graphs with billions of nodes and edges is non-trivial. Here are the core problems and standard solutions.

### The Neighbor Explosion Problem

At each GNN layer, a node aggregates from its neighbors. With k layers and average degree d, the full k-hop neighborhood has ~d^k nodes. For k=3, d=50, that's 125,000 nodes per training sample. This is the "neighborhood explosion" problem.

**Solutions:**

**Neighbor sampling (GraphSAGE):** Fix the number of neighbors sampled at each layer to a constant (e.g., 10-25). Stochastically approximates the full neighborhood. Fast, but introduces variance in gradients.

**Layer sampling (FastGCN):** Instead of sampling per-node, sample a fixed set of nodes per layer. Reduces the number of distinct node embeddings that need to be computed per batch.

**Cluster-GCN:** Partition the graph into clusters (using METIS or similar). Sample clusters for each mini-batch; nodes only aggregate from within their cluster. This makes the mini-batch a valid subgraph, eliminating inter-cluster dependencies and enabling larger batches.

```python
from torch_geometric.loader import ClusterData, ClusterLoader

# Partition graph into 1500 clusters
cluster_data = ClusterData(data, num_parts=1500, recursive=False)
train_loader = ClusterLoader(cluster_data, batch_size=20, shuffle=True)

for sub_data in train_loader:
    # sub_data is a cluster subgraph — safe to train on with no cross-cluster edges
    out = model(sub_data.x, sub_data.edge_index)
```

**SIGN (Scalable Inception Graph Neural Networks):** Pre-compute multi-hop aggregations offline. Store (A^1 X, A^2 X, ..., A^K X) as additional node features. At training time, run a simple MLP — no graph traversal needed.

```
# Pre-compute offline (once):
X_1 = A_norm @ X
X_2 = A_norm @ X_1
X_k = A_norm @ X_{k-1}

# Training: just an MLP on concatenated features
h = MLP(concat(X, X_1, X_2, ..., X_k))
```

Extremely fast training; no message passing at all during training.

### Memory Constraints

Full-batch training (computing embeddings for all nodes at once) is feasible only for small graphs (up to ~100K nodes on a GPU with 80GB RAM). For larger graphs:

- **Mini-batch training:** Process subgraphs. Requires neighbor sampling or cluster-based batching.
- **CPU offloading:** Store graph structure and features on CPU, transfer relevant subgraphs to GPU. PyTorch Geometric and DGL both support this.
- **Quantization:** Use float16 or int8 for feature storage and computation.

### Training at Billion-Scale

Production systems like PinSage (Pinterest) and Graph-based models at LinkedIn/Twitter use:

1. **Distributed training:** Partition graph across machines. Nodes near partition boundaries need cross-machine communication for their neighbor features ("cross-partition lookups").
2. **Feature caching:** Frequently accessed node features are cached in GPU memory; others are fetched from CPU or remote storage.
3. **Pre-computed embeddings:** In some systems, GNN training is replaced by iterative offline computation: generate embeddings in one pass, cache them, use them as input features for the next pass.

### Over-smoothing and Over-squashing

Two distinct depth-related failure modes:

**Over-smoothing:** With many layers, all node representations converge to the same vector (the dominant eigenvector of the normalized adjacency). Remedies: residual connections, normalization (PairNorm), jumping knowledge networks (JK-Nets).

**Over-squashing:** In graphs with long-range dependencies, information from distant nodes must "squeeze" through graph bottlenecks (low-conductance cuts, narrow paths). The gradient signal through these bottlenecks vanishes. Remedy: rewiring the graph (adding edges between distant but relevant nodes), attention mechanisms that can route information more selectively.

```python
# JK-Net: Jumping Knowledge Networks
# Concatenate representations from all layers, not just the last
class JKNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, mode='cat'):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.mode = mode
        if mode == 'cat':
            self.lin = nn.Linear(hidden_channels * num_layers, hidden_channels)

    def forward(self, x, edge_index):
        xs = []
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            xs.append(x)
        if self.mode == 'cat':
            return self.lin(torch.cat(xs, dim=-1))
        elif self.mode == 'max':
            return torch.stack(xs, dim=-1).max(dim=-1).values
```

---

## 14. Common Interview Questions

### Q1: What is the difference between GCN and GraphSAGE?

**GCN:**
- Transductive — requires the full graph at training time
- Uses symmetric normalized adjacency: D^{-1/2} A D^{-1/2}
- Aggregation is a fixed normalized mean over all neighbors
- Doesn't sample — uses all neighbors

**GraphSAGE:**
- Inductive — learns a parameterized aggregation function, applicable to new nodes
- Explicitly samples a fixed-size neighborhood at each layer
- Concatenates the node's own representation with the aggregated neighbor representation (instead of mixing them)
- Can use mean, LSTM, or max-pooling as the aggregation function

The key difference: GCN bakes the graph structure into the normalized adjacency (a fixed matrix). GraphSAGE learns an aggregation *function* that can generalize to unseen nodes/graphs.

---

### Q2: How does GAT differ from GCN? When would you prefer GAT?

GCN aggregates neighbors with weights determined entirely by graph structure (degree normalization). GAT learns to assign higher weight to more informative neighbors through a learnable attention mechanism.

**Prefer GAT when:**
- Neighbors have heterogeneous importance (some neighbors are much more informative than others)
- You want interpretable edge weights (attention scores are inspectable)
- The graph has noisy edges — attention can downweight irrelevant connections

**Prefer GCN when:**
- You want simplicity and speed (no attention computation)
- The graph is homogeneous and all neighbors are roughly equally important
- You're compute-constrained

**Gotcha:** GAT attention is computed per edge, so memory scales with |E| × num_heads × 2 × d. On dense graphs, this gets expensive.

---

### Q3: What is over-smoothing in GNNs? How do you fix it?

**What it is:** As you stack more GNN layers, the aggregation process repeatedly blends each node's representation with its neighbors. With enough iterations, this becomes equivalent to a random walk mixing, and all node representations converge to the same vector (proportional to the degree vector for GCN).

Formally: H^{(k)} = Ã^k H^{(0)} W, and as k→∞, Ã^k converges to a matrix with identical rows.

**Practical manifestation:** Node classification accuracy peaks at 2-3 layers for most benchmarks and degrades with more layers.

**Fixes:**
1. **Residual connections:** h_v^{(k)} = h_v^{(k-1)} + AGG(...)  — preserves original features
2. **Initial residual:** h_v^{(k)} = α h_v^{(0)} + (1-α) AGG(...)  — pulls back to original features (APPNP)
3. **PairNorm:** Normalizes embeddings to have zero mean and fixed norm after each layer
4. **JK-Net:** Aggregates representations from all layers, not just the final one
5. **DropEdge:** Randomly removes edges during training, reducing the number of aggregation paths and acting as regularization

---

### Q4: Explain the message passing framework. What makes MPNN general?

Message passing defines GNN computation in three steps: (1) each node sends messages to neighbors using a message function M that can depend on sender state, receiver state, and edge features; (2) each node aggregates incoming messages with a permutation-invariant AGG function; (3) each node updates its state with an update function U.

MPNN is general because M, AGG, and U can be any differentiable functions. GCN, GAT, GraphSAGE, GIN, GGNN, and most other GNNs are all special cases — they differ only in how they parameterize these three components.

---

### Q5: What is graph isomorphism and why does it matter for GNNs?

Two graphs are **isomorphic** if they're identical up to node relabeling. A key theoretical question: can GNNs distinguish non-isomorphic graphs?

The Weisfeiler-Lehman (WL) graph isomorphism test is a classical algorithm that iteratively aggregates node labels from neighbors and checks if two graphs have the same label multisets. It fails on some graph pairs (e.g., regular graphs where every node has the same degree).

**Key result (Xu et al., 2019 — GIN paper):** Standard GNN message passing with mean or max aggregation is at most as powerful as the 1-WL test. No GNN with this structure can distinguish graphs that 1-WL cannot distinguish.

**GIN (Graph Isomorphism Network)** achieves maximum discriminative power under the 1-WL constraint by using **sum aggregation** and a learnable ε parameter:

```
h_v^{(k)} = MLP^{(k)}((1 + ε^{(k)}) · h_v^{(k-1)} + Σ_{u ∈ N(v)} h_u^{(k-1)})
```

Sum aggregation is strictly more powerful than mean (which can't distinguish size of neighborhood) or max (which ignores multiplicity).

---

### Q6: How would you handle edge features in a GNN?

Several approaches:

1. **Concatenate to message:** In the message function, concatenate neighbor features with edge features: M(h_v, h_u, e_{vu}) = MLP(cat(h_u, e_{vu}))

2. **Edge-conditioned convolution (NNConv):** The weight matrix for neighbor u is parameterized by the edge feature: M(h_u, e_{vu}) = (MLP(e_{vu})) · h_u — the edge feature generates the transformation matrix

3. **Bilinear:** M = h_u^T · diag(MLP(e_{vu})) — edge feature modulates element-wise

4. **MPNN-style:** The message function in the original MPNN paper explicitly takes edge features as input alongside source/target node features

---

### Q7: How do GNNs scale to billion-node graphs?

**Three-part answer:**

1. **Neighbor sampling:** Fix the neighborhood size at each layer (GraphSAGE, PinSage). Breaks the exponential expansion of the k-hop neighborhood.

2. **Mini-batch training:** Process subgraphs rather than the full graph. Use Cluster-GCN (partition-based) or NeighborLoader (BFS-based sampling).

3. **Pre-computation (SIGN):** Compute multi-hop features offline, cache them, train a simple MLP online. No graph traversal during training.

**Production additions:** Distributed computation across machines, feature caching in GPU memory, and asynchronous stale gradient updates.

---

### Q8: What is the difference between node-level, edge-level, and graph-level tasks? How does the GNN output change?

**Node-level:** Classify each node (e.g., fraud detection, citation classification). Output: one vector per node → pass through MLP head per node.

**Edge-level:** Predict edge properties or existence (e.g., link prediction, relationship classification in KGs). Output: combine embeddings of the two endpoint nodes (concatenate or dot product) → classify.

```python
# Link prediction scoring
def link_pred_score(h_u, h_v):
    return (h_u * h_v).sum(dim=-1)  # dot product
    # OR:
    # return mlp(torch.cat([h_u, h_v], dim=-1))
```

**Graph-level:** Single output for the whole graph (e.g., molecular property prediction, graph classification). Output: pool all node embeddings to a single vector (global mean/sum/max pool or hierarchical pooling) → classify.

---

### Q9: Explain TransE and its limitations. What does RotatE fix?

**TransE:** Models relation r as a translation: h + r ≈ t. Works well for simple relational patterns (antisymmetric, one-to-one relations).

**Limitations:**
- 1-to-N relations: If entity A connects to B and C via the same relation, TransE must push emb(B) ≈ emb(C), conflating distinct entities
- Symmetric relations: (A, R, B) and (B, R, A) force r = 0
- Composition doesn't always hold

**RotatE:** Models relation as complex rotation: t = h ⊙ r with |r_i| = 1. By working in complex space with unit-modulus rotation constraints:
- Symmetry: r with phase π (so r_i = -1 for all i)
- Antisymmetry: any non-symmetric rotation
- Inversion: r₁ and r₂ = conj(r₁)
- Composition: rotate twice

---

### Q10: What happens when you run GCN on a graph with highly imbalanced node degrees (power-law degree distribution)?

The symmetric normalization D^{-1/2} A D^{-1/2} weights the contribution of neighbor u to node v by 1/sqrt(deg(u) · deg(v)).

Consequence: high-degree hub nodes contribute very little to their neighbors (their contribution is divided by their high degree). Low-degree nodes contribute more. This partially mitigates the hub dominance problem, but:

1. Hubs still see contributions from thousands of neighbors, even if individually small. Their embeddings end up as a heavily averaged representation of a huge neighborhood — information-lossy.
2. Over many layers, hubs become "bottlenecks" for information flow — everything passes through them, causing over-smoothing to concentrate around high-degree nodes first.

**Practical remedies:** Node-degree-based feature normalization, capping the number of neighbors used (like GraphSAGE), or using attention mechanisms (GAT) that can learn to ignore uninformative high-degree neighbors.

---

### Q11: How does GNN handle dynamic graphs (edges/nodes appearing over time)?

Standard GNNs are designed for static graphs. Options for dynamic graphs:

1. **Snapshot-based:** Discretize time into snapshots. Run a GNN on each snapshot; propagate the learned embeddings across time with a temporal model (GRU, LSTM, or Transformer).

2. **Continuous-time dynamic graph (CTDG):** Events happen at arbitrary timestamps. TGNN (Temporal Graph Networks) maintains a memory state per node, updated via a GRU when an event occurs involving that node. The GNN aggregation uses both current features and the node's memory state.

3. **Causal masking:** At inference time for node v, only use edges that existed before the prediction timestamp (no future leakage).

---

### Q12: How would you debug a GNN that's not training well?

Structured debugging checklist:

1. **Check node degree distribution** — if it's extremely skewed, normalization may be failing
2. **Check for over-smoothing** — are node embeddings collapsing? Compute pairwise distances across node embeddings; if they're all near-zero, you have over-smoothing. Reduce layers.
3. **Check graph connectivity** — disconnected components mean nodes in different components can never exchange information
4. **Verify the adjacency normalization** — ensure you added self-loops and symmetrically normalized
5. **Start simple** — try 1-2 GCN layers before adding complexity; verify the training loss decreases
6. **Check for data leakage** — in link prediction, ensure train/val/test splits don't share edges in a way that leaks labels
7. **Baseline comparison** — does a simple MLP on node features (ignoring graph structure) already perform well? If so, the graph structure might not be informative, or your GNN isn't using it correctly.
8. **Check gradient flow** — use gradient norms to verify backprop reaches early layers

---

## Quick Reference: Architecture Comparison

| Model | Aggregation | Inductive | Edge Features | Expressiveness | Complexity |
|---|---|---|---|---|---|
| GCN | Normalized mean | No | No | ≤ 1-WL | O(|E|d) |
| GraphSAGE | Mean/LSTM/Max | Yes | No | ≤ 1-WL | O(sample × d) |
| GAT | Learned attention | Yes | Via edge features | ≤ 1-WL | O(|E| × heads × d) |
| GIN | Sum + MLP | Yes | No | = 1-WL | O(|E|d) |
| MPNN | Arbitrary M(h,h,e) | Yes | Yes | ≤ 1-WL | Depends on M |
| GATv2 | Dynamic attention | Yes | Yes | ≤ 1-WL | O(|E| × heads × d) |

---

## Key Papers

| Paper | Year | Contribution |
|---|---|---|
| Kipf & Welling | 2017 | GCN — semi-supervised node classification |
| Hamilton et al. | 2017 | GraphSAGE — inductive learning |
| Veličković et al. | 2018 | GAT — attention over neighbors |
| Gilmer et al. | 2017 | MPNN — unifying framework |
| Xu et al. | 2019 | GIN — 1-WL expressiveness bound |
| Ying et al. | 2018 | DiffPool — hierarchical pooling |
| Bordes et al. | 2013 | TransE — translation-based KG embeddings |
| Sun et al. | 2019 | RotatE — rotation-based KG embeddings |
| Hamilton et al. | 2018 | PinSage — billion-scale recommendation |
| Brody et al. | 2022 | GATv2 — dynamic attention fix |
