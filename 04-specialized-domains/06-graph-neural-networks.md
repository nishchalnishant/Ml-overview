---
module: Specialized Domains
topic: Graph Neural Networks
subtopic: ""
status: unread
tags: [specializeddomains, ml, graph-neural-networks]
---
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

**The problem:** Standard neural networks assume fixed-size vector inputs with a clear spatial or sequential structure. Images have grid positions; text has sequence positions. CNNs exploit the pixel grid. Transformers exploit token order. But vast amounts of the world's most important data has neither structure — it is relational. It is a graph.

A drug molecule is atoms connected by bonds. The property you care about — toxicity, binding affinity — emerges from both atom identity and bonding topology. Two molecules can have identical atoms but different connectivity and behave completely differently (structural isomers). Flattening a molecule into a fixed-size vector destroys that structural information.

A social network is not a collection of user profiles. It is a system where who you know — and who they know — determines what you are exposed to. The signal for "will this user like this content?" lives not just in your profile, but in the graph structure around you.

A knowledge graph encodes facts as triples: (Barack Obama, born\_in, Hawaii). Answering multi-hop questions requires traversing edges across a structure where there is no canonical ordering and no grid.

**The core insight:** A node's representation should be a function of its own features *and* its neighbors' features. Repeat this aggregation for multiple hops to capture increasingly wide neighborhood context. The constraint that makes this well-defined: the aggregation must be *permutation invariant* — reordering a node's neighbor list must not change its representation.

**What breaks without this:** Any model that requires a fixed input size or a canonical ordering of nodes cannot handle graphs. Feeding a graph into an MLP requires choosing an arbitrary node ordering, which is not a property of the graph — it is an artifact of your representation. Two identical graphs with different orderings would produce different outputs.

---

## 2. Graph Basics

**The problem:** Before building any GNN, you need a precise vocabulary. "Node features," "neighborhoods," and "graph structure" are used loosely; without formal definitions, it is impossible to specify what the model should compute.

**The core insight:** A graph is fully described by its nodes, edges, and their features. Everything a GNN computes is a function of these three things. The adjacency matrix and Laplacian are the primary linear operators that turn graph structure into computable quantities.

**The mechanics:**

A graph is G = (V, E) where:
- **V** = set of nodes (vertices), |V| = N
- **E** ⊆ V × V = set of edges

Each node v ∈ V may have a feature vector **x**\_v ∈ R^d. Each edge (u, v) ∈ E may have a feature vector **e**\_{uv}.

### Adjacency Matrix

The adjacency matrix **A** ∈ {0,1}^{N×N}:

```
A[i][j] = 1  if edge (i, j) exists
A[i][j] = 0  otherwise
```

For weighted graphs, A[i][j] holds the edge weight. For undirected graphs, **A** is symmetric. For directed graphs, it generally is not.

### Degree

The **degree** of a node is the number of edges connected to it.

- Undirected: deg(v) = Σ\_j A[v][j]
- Directed: in-degree = Σ\_j A[j][v], out-degree = Σ\_j A[v][j]

The **degree matrix** D is diagonal where D[i][i] = deg(i). It appears in every GCN derivation.

### Structural Vocabulary

| Concept | Definition | Why it matters |
|---|---|---|
| Neighborhood N(v) | All nodes directly connected to v | Defines the scope of one GNN layer |
| Path | Sequence of edges connecting two nodes | Determines information flow depth |
| Connected component | Maximal subgraph where all nodes are reachable | Nodes in different components can never exchange information |
| Diameter | Longest shortest path in the graph | Bounds how many layers you need to connect any two nodes |
| Clustering coefficient | Fraction of a node's neighbors that are also connected to each other | Measures local "cliqueness" |

### The Laplacian

**The problem:** You need a linear operator on graph signals that encodes local smoothness — how much a signal changes between adjacent nodes.

**The core insight:** The graph Laplacian L = D - A is the discrete analogue of the continuous Laplace operator. Its eigenvectors form a Fourier basis for the graph, with small eigenvalues corresponding to smooth signals and large eigenvalues to signals that change sharply between adjacent nodes.

The **normalized Laplacian**:

```
L_norm = D^{-1/2} L D^{-1/2} = I - D^{-1/2} A D^{-1/2}
```

Its eigenvalues lie in [0, 2]. Spectral GNN methods work directly with these eigenvectors.

**What breaks:** The eigenvectors of L change whenever you add or remove a single node. This makes spectral methods fundamentally transductive — the learned filter is tied to the specific graph it was trained on.

---

## 3. Challenges with Graphs

**The problem:** Even if you accept that graphs are the right representation, three structural properties make them incompatible with standard neural network machinery.

**The core insight:** Each property requires a specific architectural response. Variable size requires parameter sharing across nodes. No canonical ordering requires permutation invariance. Variable-degree neighborhoods require invariant aggregation. GNNs are architectures that satisfy all three simultaneously.

### Variable Size

A molecular graph has 20–50 nodes. A social graph has billions. Unlike images (pad/crop to 224×224) or text (pad to 512 tokens), there is no canonical "resize" operation for graphs that preserves structural meaning. Any architecture must handle arbitrary N.

### No Natural Ordering

Relabeling the nodes of a graph — swap node 3 and node 7 — changes nothing about the graph. But it changes the feature matrix. Any model must be **permutation invariant** (graph-level output unchanged by relabeling) or **permutation equivariant** (node-level output transforms consistently with relabeling):

```
f(PX, PAP^T) = f(X, A)   for any permutation matrix P
```

This rules out naive MLP or RNN approaches that depend on a fixed ordering.

**What breaks:** If you feed adjacency rows as sequential input to an RNN, the model learns dependencies tied to node indices, not graph structure. Two isomorphic graphs with different node orderings will produce different outputs — a fundamental failure of representation.

### Variable-Degree Aggregation

Different nodes have different numbers of neighbors. Your aggregation function must produce a fixed-size output from a variable-size input while being permutation invariant. Sum, mean, and max are all permutation invariant; concatenation is not.

**What breaks with bad aggregation:** Mean pooling over neighbors cannot distinguish a node with two neighbors [A, B] from a node with 100 neighbors that happen to have the same mean embedding. The structural signal — that one node has a much larger neighborhood — is lost.

---

## 4. Graph Convolutional Networks (GCN)

**The problem:** You need an operation that aggregates information from a node's neighborhood in a permutation-invariant way, shares parameters across all nodes so the model scales regardless of graph size, and can be stacked to reach deeper neighborhoods.

**The core insight:** Treat the normalized adjacency matrix as a fixed linear operator and apply it to the node feature matrix. This computes a weighted average of each node's neighbors' features in one matrix multiplication. Stack this with a learnable projection and a nonlinearity, and you have a GNN layer.

**The mechanics (Kipf & Welling, 2017):**

```
H^{(k+1)} = σ( Ã H^{(k)} W^{(k)} )
```

Where:
- H^{(k)} ∈ R^{N×d\_k} = node feature matrix at layer k
- W^{(k)} ∈ R^{d\_k × d\_{k+1}} = learnable weight matrix (shared across all nodes)
- σ = nonlinearity (ReLU)
- Ã = D̃^{-1/2} (A + I) D̃^{-1/2} = symmetrically normalized adjacency with self-loops

**Why self-loops?** Without A + I, a node's own representation at layer k does not feed into its representation at layer k+1 — the update uses only neighbors, not the node itself. Adding I fixes this: each node is its own neighbor.

**Why the D̃^{-1/2} normalization?** Without normalization, high-degree nodes aggregate much larger signals than low-degree nodes, and outputs explode for hubs. The symmetric normalization keeps magnitudes stable: the contribution of neighbor u to node v is scaled by 1/√(deg(u)·deg(v)).

### The Message Passing Framework

The GCN layer is a specific instance of a more general pattern:

**Step 1 — Message:** For every edge (u, v), compute a message from u to v:
```
m_{u→v}^{(k)} = MSG^{(k)}(h_u^{(k-1)}, h_v^{(k-1)}, e_{uv})
```

**Step 2 — Aggregate + Update:** For each node v, aggregate the incoming messages and update its embedding:
```
h_v^{(k)} = UPDATE^{(k)}(h_v^{(k-1)}, AGG({m_{u→v}^{(k)} : u ∈ N(v)}))
```

After k layers, h\_v^{(k)} encodes information from the k-hop neighborhood of v.

### Aggregation Variants

**Mean aggregation:**
```
h_v^{(k)} = σ( W · MEAN({h_u^{(k-1)} : u ∈ N(v) ∪ {v}}) )
```
Ignores neighborhood size. Cannot distinguish a node with 2 neighbors [A, B] from one with 100 neighbors with the same mean.

**Sum aggregation:**
```
h_v^{(k)} = σ( W · SUM({h_u^{(k-1)} : u ∈ N(v)}) )
```
Sensitive to degree. Can distinguish neighborhood sizes — if all neighbors have distinct features, sum is injective. GIN proves sum is strictly more expressive than mean.

**Max aggregation:**
```
h_v^{(k)} = σ( W · MAX({h_u^{(k-1)} : u ∈ N(v)}) )
```
Captures the "most extreme" feature value across neighbors. Loses information about how many neighbors hold that value.

### Layer-by-Layer Neighborhood Expansion

- 1 layer: each node sees its direct neighbors (1-hop)
- 2 layers: each node sees neighbors of neighbors (2-hop)
- k layers: each node sees its entire k-hop neighborhood

For a molecule, 2–3 layers typically covers all chemically relevant context. For social graphs, 6 hops technically reaches most of the graph (six degrees of separation) — but the 6-hop neighborhood can be the entire graph, which leads to over-smoothing.

### PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x, adj_norm):
        # adj_norm: pre-computed D^{-1/2} (A + I) D^{-1/2}
        support = self.linear(x)                   # (N, out_features)
        out = torch.sparse.mm(adj_norm, support)   # (N, out_features)
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
        return F.log_softmax(x, dim=1)
```

```python
import scipy.sparse as sp
import numpy as np

def normalize_adj(adj):
    """Symmetric normalization: D^{-1/2} (A + I) D^{-1/2}"""
    adj = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv_sqrt = np.power(rowsum, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat = sp.diags(d_inv_sqrt)
    return d_mat @ adj @ d_mat
```

**What breaks:**

1. **Transductive only:** The normalized adjacency is computed for the full training graph. A new node arriving at inference time requires recomputing the full normalization. This is the motivation for GraphSAGE.
2. **Over-smoothing:** Stack too many layers and all node representations converge to the same vector (analogous to a random walk mixing). Empirically, 2–3 layers is often optimal.
3. **Fixed equal weights for all neighbors:** A node's close friends and casual acquaintances get the same weight. GAT addresses this.

---

## 5. GraphSAGE — Inductive Learning

**The problem:** Standard GCN computes embeddings using the full graph adjacency. This is **transductive** — you can only generate embeddings for nodes present during training. New users join a social network every day. A graph with 1B nodes does not fit in memory. You want a model that generalizes to unseen graph structures without retraining.

**The core insight:** Instead of baking the graph structure into a fixed normalized matrix, learn a *function* — a parameterized aggregator — that can be applied to any node's neighborhood regardless of whether those nodes were seen during training. The function is what gets learned; the graph is just input to that function.

**The mechanics (Hamilton et al., 2017):**

For each layer k = 1 ... K, for each node v:

```
1. Sample a fixed-size neighborhood: S_v ⊆ N(v)
2. Aggregate neighbor representations:
     h_N(v)^k = AGG_k({h_u^{k-1} : u ∈ S_v})
3. Concatenate with own representation and transform:
     h_v^k = σ( W^k · CONCAT(h_v^{k-1}, h_N(v)^k) )
4. L2-normalize:
     h_v^k = h_v^k / ||h_v^k||_2
```

**Inference on a new node:** Given features x\_v and observed neighbors, run the same aggregator function. No retraining, no recomputing global normalization.

### Why Neighborhood Sampling?

Without sampling, the computational graph for a single node at depth k grows as |N(v)|^k. With average degree 200 and k=2, that is 40,000 nodes per training example; at k=3, 8 million. Sampling fixes the receptive field to a constant — typically 25 neighbors at hop 1, 10 at hop 2.

### Aggregator Variants

**Mean:**
```python
h_v^k = σ( W · MEAN([h_v^{k-1}] ∪ {h_u^{k-1} : u ∈ S_v}) )
```

**LSTM:** Shuffle neighbors randomly, feed through LSTM, take final hidden state. Not permutation invariant by design, but random shuffling approximates it. Works well in practice despite the theoretical gap.

**Max pooling:**
```python
h_N(v) = MAX({σ(W_pool · h_u^{k-1} + b) : u ∈ S_v})
```

### Unsupervised Training Objective

```
J(z_v) = -log(σ(z_v^T z_u)) - Q · E_{v_n ~ P_n}[log(σ(-z_v^T z_{v_n}))]
```

u is a neighbor of v (positive pair), v\_n is a random negative sample, Q is the number of negatives.

### PyTorch Implementation

```python
from torch_geometric.nn import SAGEConv
from torch_geometric.loader import NeighborLoader

class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        return self.conv2(x, edge_index)

loader = NeighborLoader(
    data,
    num_neighbors=[25, 10],
    batch_size=1024,
    input_nodes=data.train_mask,
)
```

**What breaks:**

**Sampling variance:** Stochastically approximating the full neighborhood introduces variance in gradients. Different runs sample different neighborhoods, making training less deterministic.

**LSTM aggregator's permutation issue:** The LSTM processes a sequence; random shuffling is an approximation of permutation invariance, not the real thing. If neighbor orderings happen to be non-random, the LSTM learns spurious ordering biases.

---

## 6. Graph Attention Networks (GAT)

**The problem:** GCN aggregates neighbors with weights determined entirely by graph structure (degree normalization). Every neighbor contributes equally, scaled only by degree. But in a social network, your best friend's opinion should matter more than a spam account that followed you. In a citation graph, a directly relevant paper should matter more than a tangential one.

**The core insight:** Instead of computing aggregation weights from degree alone, learn them from the content of the node features. Pairs of nodes that are semantically compatible should have high attention weight; incompatible pairs should have low weight.

**The mechanics (Veličković et al., 2018):**

**Step 1:** Project node features:
```
z_v = W h_v
```

**Step 2:** Compute unnormalized attention between v and each neighbor u:
```
e_{vu} = LeakyReLU( a^T · CONCAT(z_v, z_u) )
```
**a** ∈ R^{2d'} is a learnable attention vector.

**Step 3:** Normalize with softmax over v's neighborhood:
```
α_{vu} = exp(e_{vu}) / Σ_{w ∈ N(v) ∪ {v}} exp(e_{vw})
```

**Step 4:** Weighted aggregation:
```
h_v' = σ( Σ_{u ∈ N(v) ∪ {v}} α_{vu} · W h_u )
```

**Multi-head attention:** Multiple independent attention mechanisms, concatenated (intermediate layers) or averaged (final layer):
```
h_v' = CONCAT_{k=1}^{K} σ( Σ_{u ∈ N(v)} α_{vu}^k · W^k h_u )
```

### Connection to Transformer Attention

GAT attention is transformer attention restricted to the graph adjacency:
- Query = node v through W\_Q
- Key = neighbor u through W\_K
- Value = W\_V h\_u
- Mask = graph adjacency (only attend to actual neighbors)

The difference: transformers attend over all pairs (full attention); GAT attends only within each node's neighborhood (sparse, structured attention). This makes GAT much more efficient on graphs where neighborhoods are small relative to graph size.

**GATv2 (Brody et al., 2022)** fixes a subtle expressiveness problem in the original: in GATv1, the ranking of attention scores cannot depend on the query node for the same set of keys (the attention is "static"). GATv2 changes the order of operations so attention is fully dynamic.

### PyTorch Implementation

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
        return F.log_softmax(self.conv2(x, edge_index), dim=1)
```

**What breaks:**

**Memory cost:** Attention is computed per edge. Memory scales as O(|E| × heads × d). On dense graphs or graphs with high-degree hubs, this becomes prohibitive.

**Attention doesn't fix over-smoothing:** GAT learns which neighbors to weight more, but with many layers, weighted averaging still converges. Attention buys you better aggregation at each layer, not a cure for depth pathologies.

---

## 7. Message Passing Neural Networks (MPNN)

**The problem:** GCN, GraphSAGE, and GAT each solve specific problems with specific aggregation choices. But they share a common structure. Is there a single framework that unifies them all and makes the design space explicit?

**The core insight:** Any GNN that operates by aggregating local neighborhood information can be expressed in three primitives: a message function M (how each neighbor sends information), an aggregate function AGG (how messages are combined), and an update function U (how the current state is revised). Different GNNs are different choices for these three functions.

**The mechanics (Gilmer et al., 2017):**

**Message passing phase** (T steps):
```
m_v^{t+1} = Σ_{w ∈ N(v)} M_t(h_v^t, h_w^t, e_{vw})
```

**Update phase:**
```
h_v^{t+1} = U_t(h_v^t, m_v^{t+1})
```

**Readout phase** (for graph-level tasks):
```
ŷ = R({h_v^T : v ∈ G})
```

### How Other Models Fit In

| Model | Message function M\_t | Update function U\_t |
|---|---|---|
| GCN | h\_w (neighbor features) | Linear + ReLU |
| GraphSAGE | h\_w | Linear(concat(h\_v, mean(neighbors))) |
| GAT | α\_{vw} · W h\_w | Sum of weighted messages |
| GGNN | h\_w | GRU |

### Edge Features as First-Class Citizens

The MPNN framing makes edge features explicit. In chemistry, bond type (single, double, aromatic) is critical. The message function takes both the neighbor's state and the edge type:

```python
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

This allows information to flow over multiple steps while gating out irrelevant information, analogous to sequence models.

**What breaks:**

**MPNN is a framework, not a solution:** Choosing M, AGG, and U well still requires domain knowledge. A poorly chosen M that ignores edge features, or an AGG that is not permutation invariant, still fails. The framework makes the design space explicit — it does not fill it in.

**The 1-WL ceiling:** Regardless of how M, AGG, and U are parameterized, standard MPNN-style GNNs cannot distinguish graphs that the 1-Weisfeiler-Lehman graph isomorphism test cannot distinguish. This is a fundamental expressiveness limit of the message-passing paradigm (proved by Xu et al., 2019).

---

## 8. Spectral vs Spatial Graph Convolutions

**The problem:** "Convolution" on a grid (images) is well-defined because the grid has a regular structure that makes the Fourier transform natural. Graphs have no grid structure. What does convolution even mean on an arbitrary graph?

**The core insight:** Two distinct answers give rise to two families of GNNs. Spectral methods define convolution via the graph's Fourier transform (Laplacian eigenvectors). Spatial methods define it directly as local neighborhood aggregation, bypassing eigendecompositions entirely.

### Spectral Approach

**The core insight:** Define convolution in the frequency domain via the graph Fourier transform. The Laplacian eigenvectors play the role of Fourier basis vectors on a graph.

Eigenvector decomposition: L = U Λ U^T
- U = matrix of eigenvectors (Fourier basis)
- Λ = diagonal matrix of eigenvalues (frequencies)

Graph Fourier transform of signal x: x̂ = U^T x

Spectral convolution with filter g:
```
x *_G g = U · g_θ(Λ) · U^T x
```

**What breaks immediately:** Computing U requires full eigendecomposition — O(N^2) computation and O(N^2) memory. Not scalable.

**ChebNet (Defferrard et al., 2016)** approximates the filter with Chebyshev polynomials of degree K:
```
g_θ(L) ≈ Σ_{k=0}^{K} θ_k T_k(L̃)
```
This makes computation O(K|E|) — linear in the number of edges.

**GCN simplifies further:** Take K=1, assume λ\_max ≈ 2, and the GCN formula drops out. So GCN is a first-order Chebyshev approximation of a spectral filter. The spectral framework motivates GCN; the spatial view makes it usable.

### Spatial Approach

**The core insight:** Forget eigendecompositions. Directly define how a node aggregates information from its spatial (graph-neighborhood) neighbors. GraphSAGE, GAT, GIN, and most modern methods take this approach.

**Why spatial dominates in practice:**
- Naturally inductive — parameters shared across all nodes, applicable to new graphs
- Does not require recomputing eigendecompositions for new graph sizes
- Edge features are easy to incorporate into the message function
- Scales to billion-node graphs with neighborhood sampling

**Why spectral has niche value:**
- Theoretically grounded in signal processing
- Can learn global frequency patterns
- Principled way to think about "smoothness" of node representations
- Useful for fixed-topology graphs (brain connectivity, sensor networks)

**What breaks with spectral at scale:** The eigenvectors of the Laplacian change when you add or remove a single node. This makes spectral methods fundamentally transductive — the learned filter is tied to the specific graph it was trained on.

---

## 9. Graph Pooling

**The problem:** A single graph often represents one data point — a molecule, a scene graph, a social cluster. Node-level embeddings give you one vector per node. You need a single vector for the whole graph. How do you compress a variable-size set of node embeddings into one fixed-size representation without losing structural information?

**The core insight:** Global pooling (sum/mean/max over all node embeddings) is fast and permutation invariant but throws away all information about which nodes are central or how the graph is organized hierarchically. Hierarchical pooling learns to coarsen the graph progressively, analogous to pooling in CNNs.

### Global Pooling

**Mean pooling:**
```
h_G = (1/|V|) Σ_{v ∈ V} h_v
```
Invariant to graph size. Cannot distinguish graphs that have the same node mean but different total "mass" or different structural arrangement.

**Sum pooling:**
```
h_G = Σ_{v ∈ V} h_v
```
More expressive than mean — can distinguish graphs with the same mean but different sizes.

**Max pooling:**
```
h_G = MAX_{v ∈ V}({h_v})  (element-wise max)
```
Captures the most prominent feature across all nodes.

```python
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool

graph_embedding = global_mean_pool(node_embeddings, batch)
```

### Hierarchical Pooling: DiffPool

**The problem global pooling doesn't solve:** It treats all nodes as equally important and ignores how the graph is organized into communities, modules, or hierarchical structures.

**DiffPool (Ying et al., 2018):** At each pooling layer, learn a soft cluster assignment matrix S ∈ R^{N×C} mapping N nodes to C clusters:
```
S^(l) = softmax( GNN_pool^(l)(A^(l), X^(l)) )
```

Pooled node features and adjacency:
```
X^(l+1) = S^(l)^T · Z^(l)
A^(l+1) = S^(l)^T · A^(l) · S^(l)
```

Two auxiliary losses encourage good clustering:
1. **Link prediction loss:** adjacent nodes should be in the same cluster
2. **Entropy regularization:** cluster assignments should be sharp (near-binary), not fuzzy

**What breaks with DiffPool:** O(N^2) memory for the assignment matrix. For large graphs, this is prohibitive.

### Alternatives

- **MinCutPool:** Spectral clustering loss for cluster assignments
- **Top-K Pooling:** Learn a scalar importance score per node; keep the top k, drop the rest
- **SAGPooling:** Use graph attention scores to rank and select nodes

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

### Molecular Property Prediction

**The problem:** Given a drug candidate molecule, predict properties: solubility, toxicity, binding affinity, ADMET. Traditional approaches require hand-crafted molecular fingerprints (Morgan fingerprints, MACCS keys) that encode structural patterns manually.

**The core insight:** Molecules are graphs — atoms are nodes (features: atomic number, charge, hybridization), bonds are edges (features: bond type, aromatic). Molecular properties are local — a functional group's contribution depends on the nearby chemical environment, not the entire molecule. This maps directly to the local neighborhood aggregation of GNNs. The GNN discovers which structural patterns matter; you do not need to specify them in advance.

**What breaks with fingerprints:** Fingerprints encode a fixed vocabulary of substructures. If the relevant substructure is not in the vocabulary, it is invisible to the model. GNNs can learn to detect novel structural motifs directly from data.

```python
from torch_geometric.nn import NNConv, global_add_pool

class MoleculeGNN(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, out_dim):
        super().__init__()
        edge_nn1 = nn.Linear(edge_dim, node_dim * hidden_dim)
        self.conv1 = NNConv(node_dim, hidden_dim, edge_nn1)
        edge_nn2 = nn.Linear(edge_dim, hidden_dim * hidden_dim)
        self.conv2 = NNConv(hidden_dim, hidden_dim, edge_nn2)
        self.readout = nn.Linear(hidden_dim, out_dim)

    def forward(self, data):
        x, edge_index, edge_attr, batch = (
            data.x, data.edge_index, data.edge_attr, data.batch
        )
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x = global_add_pool(x, batch)
        return self.readout(x)
```

### Recommendation Systems — Pinterest PinSage

**The problem:** Pinterest has 3B nodes and 18B edges. A user saves a pin to a board; that's an edge. Two pins of coffee mugs might look identical visually but have completely different contexts — one is on a "home brewing" board, the other on a "party supplies" board. Raw image embeddings miss this contextual signal entirely.

**The core insight:** Graph context encodes usage patterns that raw features miss. Nodes that are co-engaged by similar users cluster together in the graph even when their raw features diverge. Multi-hop graph traversal exposes second-order co-engagement patterns.

**PinSage contributions:**
1. **Random walk-based neighbor sampling:** Instead of uniform sampling, use biased random walks — nodes visited more frequently are more important neighbors. This approximates personalized PageRank.
2. **Importance-based aggregation:** Assign weights to sampled neighbors proportional to visit frequency.
3. **Curriculum training:** Start with easy negatives (random items), progressively use harder negatives.

### Knowledge Graphs and Entity Disambiguation

**The problem:** Given the mention "Washington" in text, determine whether it refers to the city, the state, George Washington, or Denzel Washington. Text context alone is often ambiguous. The knowledge graph around each candidate entity provides rich structural context that disambiguates.

**The GNN approach:** Build a local subgraph around each entity candidate (its neighbors in the KG, their types, their connections). Run a GNN to get a context-aware embedding for each candidate. The candidate whose embedding is most compatible with the textual context wins.

### Fraud Detection

**The problem:** In financial transaction graphs, fraudsters form tightly-connected communities because they control the accounts on both sides of synthetic transactions. Node-level features alone (account age, transaction velocity) can be gamed. Structural signals are harder to fake.

**The core insight:** Fraud signals propagate through the graph. If your neighbors are flagged as fraudulent, your risk score should increase. GNNs aggregate this neighborhood signal across multiple hops — to game the system, a fraudster would need to also control a large connected neighborhood.

```python
class FraudGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels)
        self.classifier = nn.Linear(hidden_channels, 2)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        return self.classifier(x)
```

**What breaks:** Class imbalance — fraudulent transactions are typically 0.1% of all transactions. Strategies: weighted cross-entropy, oversampling fraud nodes, focal loss.

---

## 11. Knowledge Graph Embeddings

**The problem:** Knowledge graphs like Wikidata represent facts as triples (head, relation, tail): (Paris, capital\_of, France). KGs are always incomplete — facts exist in the world that have not been entered. Given the triples that are known, how do you predict the missing ones? You need embeddings where valid triples score high and invalid triples score low.

**The core insight:** Assign each entity and each relation a low-dimensional vector. Define a scoring function on triples. Train by maximizing the score of observed triples relative to corrupted (invalid) triples.

### TransE

**The core insight:** Model a relation as a translation in embedding space. If (h, r, t) is a valid triple, then **h** + **r** ≈ **t**. Relations are vectors that move you from one entity to another.

**Score function:**
```
f(h, r, t) = -||h + r - t||
```

**Training:**
```
L = Σ_{(h,r,t)∈T} Σ_{(h',r,t')∉T} max(0, γ + f(h,r,t) - f(h',r,t'))
```

**What breaks:**
- 1-to-N relations: If entity A has many tail entities via the same relation (many cities connected to a country by "has\_city"), TransE must push all tail embeddings to the same position, conflating distinct entities.
- Symmetric relations: (A, sibling\_of, B) and (B, sibling\_of, A) require **r** = **0**.

### RotatE

**The core insight:** Model relations as rotations in complex space. Valid triples satisfy **t** = **h** ⊙ **r** (element-wise complex multiplication) with |r\_i| = 1 — each dimension rotates, it does not scale. Symmetry becomes a rotation by π; inversion becomes the conjugate relation.

**Score function:**
```
f(h, r, t) = -||h ⊙ r - t||
```

**Why RotatE is more expressive:**

| Relation pattern | TransE | RotatE |
|---|---|---|
| Symmetry (A↔B) | Cannot model | r\_i = ±1 (rotation by 0 or π) |
| Antisymmetry | Yes | Yes |
| Inversion (r₁ = r₂⁻¹) | Cannot model | r₁ ⊙ r₂ = 1 |
| Composition (r₁ ∘ r₂ = r₃) | Partially | Yes |

```python
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
        return F.relu(self.margin + neg_score - pos_score).mean()
```

### Other Notable KG Embedding Methods

- **DistMult:** Score = h^T diag(r) t — bilinear, symmetric (cannot model asymmetric relations)
- **ComplEx:** Complex-valued DistMult; handles asymmetric relations
- **RGCN:** Separate weight matrices per relation type in the GCN update — bridges KG embeddings and GNNs
- **CompGCN:** Jointly embeds entities and relations using composition operations

---

## 12. GNNs in LLMs

**The problem:** LLMs are trained to predict tokens, not to reason over structured relational data. Multi-hop questions like "Who is the maternal grandfather of the US president who succeeded Nixon?" require chaining (Nixon → Ford → Ford's mother → Ford's maternal grandfather) — a path traversal on a knowledge graph. LLMs can answer this by memorizing the path at training time, but they fail on facts they have not seen or on complex compositional reasoning that requires consistent state tracking.

**The core insight:** GNNs and LLMs are complementary. LLMs are good at language understanding and generation; GNNs are good at structured relational reasoning. Combining them lets you ground language generation in structured knowledge.

### GraphRAG

**The problem:** Vanilla RAG retrieves chunks that are locally relevant but misses global structure. "What are the main themes in this corpus?" requires understanding relationships between topics across many documents — not just finding locally relevant passages.

**Microsoft's GraphRAG (2024):**
1. Index a document corpus by building a knowledge graph from extracted entities and relationships
2. Detect communities (clusters of related entities) in the graph
3. At query time, retrieve relevant graph communities, summarize each community's subgraph
4. Pass summaries to the LLM

**Why this beats chunk retrieval for global questions:** Community-based retrieval surfaces patterns that span the entire graph. Chunk retrieval only surfaces locally relevant passages.

### LLM + GNN Hybrid

**Text-attributed graphs:** Use an LLM to generate rich embeddings for each node (product descriptions, paper abstracts) as initial node features, then run a GNN to propagate structural information. Combines language understanding with graph structure.

**G-Retriever (2024):** Given a question over a knowledge graph, retrieves a prize-collecting Steiner tree (the minimal subgraph connecting all relevant entities), encodes it with a GNN, then uses the GNN embedding as soft prompt tokens for an LLM.

```python
from transformers import AutoModel
from torch_geometric.nn import GCNConv

class TextGNN(nn.Module):
    def __init__(self, lm_model_name, hidden_dim, num_classes):
        super().__init__()
        self.lm = AutoModel.from_pretrained(lm_model_name)
        lm_dim = self.lm.config.hidden_size
        self.projection = nn.Linear(lm_dim, hidden_dim)
        self.gnn1 = GCNConv(hidden_dim, hidden_dim)
        self.gnn2 = GCNConv(hidden_dim, num_classes)

    def forward(self, input_ids, attention_mask, edge_index):
        lm_out = self.lm(input_ids=input_ids, attention_mask=attention_mask)
        x = lm_out.last_hidden_state[:, 0, :]  # CLS token
        x = self.projection(x)
        x = F.relu(self.gnn1(x, edge_index))
        return self.gnn2(x, edge_index)
```

**What breaks:**

**Scale mismatch:** LLMs operating on text can process thousands of tokens. GNNs operating on graphs can handle millions of nodes. Bridging these — encoding a large graph's structure into a form an LLM can use — is an active open problem.

**Serialization loses structure:** Representing a graph as text (a list of triples) loses the structural inductive biases that make GNNs effective. But injecting GNN embeddings into an LLM requires aligning two different representation spaces.

---

## 13. Scalability Challenges

**The problem:** Computing embeddings for a single node at depth k requires the embeddings of its k-hop neighborhood. With average degree d and k layers, that is ~d^k nodes per training sample. At k=3 and d=50, that is 125,000 nodes per example. The adjacency matrix for a 1B-node graph has 10^18 entries, of which only ~2×10^11 are non-zero.

### The Neighbor Explosion Problem

**Neighbor sampling (GraphSAGE):** Fix the number of neighbors sampled at each layer (e.g., 10–25). Stochastically approximates the full neighborhood. Fast, but introduces gradient variance.

**Layer sampling (FastGCN):** Instead of sampling per-node, sample a fixed set of nodes per layer. Reduces the number of distinct node embeddings per batch.

**Cluster-GCN:** Partition the graph into clusters (METIS or similar). Sample clusters for each mini-batch; nodes only aggregate from within their cluster. Eliminates inter-cluster dependencies and enables larger batches.

```python
from torch_geometric.loader import ClusterData, ClusterLoader

cluster_data = ClusterData(data, num_parts=1500, recursive=False)
train_loader = ClusterLoader(cluster_data, batch_size=20, shuffle=True)
```

**SIGN (Scalable Inception GNN):** Pre-compute multi-hop aggregations offline. Store (A^1 X, A^2 X, ..., A^K X) as additional node features. At training time, run a simple MLP — no graph traversal during training at all.

```
# Pre-compute offline (once):
X_1 = A_norm @ X
X_2 = A_norm @ X_1
...

# Training: just an MLP on concatenated features
h = MLP(concat(X, X_1, X_2, ..., X_k))
```

### Over-smoothing and Over-squashing

**Over-smoothing:** With many layers, all node representations converge to the same vector. The normalized adjacency Ã is a contractive operator; repeatedly applying it drives node representations toward the dominant eigenvector (proportional to the degree vector for GCN). Empirically, node classification accuracy typically peaks at 2–3 layers.

**What breaks:** Node representations that are all nearly identical cannot distinguish nodes, making the model useless for any node-level task.

**Fixes:** Residual connections, initial residual (APPNP: blend each layer with the original features), PairNorm, JK-Net (jump-connect all layers).

**Over-squashing:** In graphs with long-range dependencies, information from distant nodes must flow through narrow graph bottlenecks (low-conductance cuts). The gradient signal through these bottlenecks vanishes.

**What breaks:** Tasks that require coordinating distant parts of the graph — e.g., knowing that the two endpoints of a long chain are connected — are essentially unlearnable by standard message-passing GNNs.

**Fixes:** Graph rewiring (add edges between distant but relevant nodes), attention mechanisms that route information selectively.

```python
# JK-Net: Jumping Knowledge Networks
class JKNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, mode='cat'):
        super().__init__()
        self.convs = nn.ModuleList([GCNConv(in_channels, hidden_channels)])
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
- Normalized adjacency D^{-1/2} A D^{-1/2} is a fixed matrix computed once
- Aggregation is a fixed normalized mean over all neighbors
- Cannot generalize to new nodes

**GraphSAGE:**
- Inductive — learns a parameterized aggregation function applicable to new nodes
- Samples a fixed-size neighborhood at each layer
- Concatenates the node's own representation with the aggregated neighbor representation
- Can use mean, LSTM, or max-pooling as aggregation

The key difference: GCN bakes the graph structure into a fixed matrix. GraphSAGE learns an aggregation *function* that generalizes to unseen nodes and graphs.

---

### Q2: How does GAT differ from GCN? When would you prefer GAT?

GCN aggregates neighbors with weights determined entirely by degree normalization. GAT learns to assign higher weight to more informative neighbors through a learnable attention mechanism.

**Prefer GAT when:**
- Neighbors have heterogeneous importance
- You want interpretable edge weights (attention scores are inspectable)
- The graph has noisy edges — attention can downweight irrelevant connections

**Prefer GCN when:**
- You want simplicity and speed (no attention computation)
- The graph is homogeneous and all neighbors are roughly equally important
- Memory is constrained (GAT memory scales with |E| × heads × d)

---

### Q3: What is over-smoothing? How do you fix it?

**What it is:** As you stack more GNN layers, repeated neighborhood averaging blends node representations together until they all converge to the same vector. Formally: as k→∞, Ã^k H^{(0)} converges to a matrix with identical rows.

**Practical manifestation:** Node classification accuracy peaks at 2–3 layers and degrades with more layers.

**Fixes:**
1. **Residual connections:** h\_v^{(k)} = h\_v^{(k-1)} + AGG(...)
2. **Initial residual (APPNP):** h\_v^{(k)} = α h\_v^{(0)} + (1-α) AGG(...) — pulls back toward original features
3. **PairNorm:** Normalizes embeddings to have zero mean and fixed norm after each layer
4. **JK-Net:** Aggregates representations from all layers, not just the final one
5. **DropEdge:** Randomly removes edges during training, reducing aggregation paths

---

### Q4: What is the message passing framework? What makes MPNN general?

Message passing defines GNN computation in three steps: (1) each node sends messages to neighbors using a message function M that can depend on sender state, receiver state, and edge features; (2) each node aggregates incoming messages with a permutation-invariant AGG function; (3) each node updates its state with an update function U.

MPNN is general because M, AGG, and U can be any differentiable functions. GCN, GAT, GraphSAGE, GIN, GGNN are all special cases — they differ only in how they parameterize these three components.

---

### Q5: What is the Weisfeiler-Lehman test? Why does it matter for GNNs?

The WL test is a classical graph isomorphism algorithm that iteratively assigns color labels to nodes by hashing (node's label, multiset of neighbor labels). Two graphs are distinguished if their label distributions ever differ.

**Key result (Xu et al., 2019 — GIN paper):** Standard GNN message passing with mean or max aggregation is at most as powerful as the 1-WL test. No GNN with this structure can distinguish graphs that 1-WL cannot distinguish.

This matters because 1-WL fails on some non-isomorphic graphs — for example, two regular graphs of different structure where every node has the same degree. GNNs that use these aggregation schemes inherit this blindspot.

**GIN** achieves maximum discriminative power within the 1-WL constraint by using **sum aggregation** (strictly more expressive than mean or max) and a learnable ε parameter:
```
h_v^{(k)} = MLP^{(k)}((1 + ε^{(k)}) · h_v^{(k-1)} + Σ_{u ∈ N(v)} h_u^{(k-1)})
```

---

### Q6: How would you handle edge features in a GNN?

Three main approaches:

1. **Concatenate to message:** M(h\_v, h\_u, e\_{vu}) = MLP(cat(h\_u, e\_{vu}))

2. **Edge-conditioned convolution (NNConv):** The weight matrix for neighbor u is parameterized by the edge feature: M(h\_u, e\_{vu}) = MLP(e\_{vu}) · h\_u — the edge feature generates the transformation matrix

3. **Bilinear:** M = h\_u^T · diag(MLP(e\_{vu})) — edge feature modulates element-wise

---

### Q7: How do GNNs scale to billion-node graphs?

Three-part answer:

1. **Neighbor sampling (GraphSAGE, PinSage):** Fix the neighborhood size at each layer. Breaks the exponential expansion of the k-hop neighborhood.

2. **Mini-batch training (Cluster-GCN, NeighborLoader):** Process subgraphs. Partition-based batching eliminates cross-cluster dependencies.

3. **Pre-computation (SIGN):** Compute multi-hop features offline, cache them, train a simple MLP online. No message passing during training.

Production additions: distributed computation across machines, feature caching in GPU memory, asynchronous stale gradient updates.

---

### Q8: What is the difference between node-level, edge-level, and graph-level tasks?

**Node-level:** Classify each node (fraud detection, citation classification). Output: one vector per node → MLP head per node.

**Edge-level:** Predict edge properties or existence (link prediction, KG relation classification). Output: combine embeddings of the two endpoint nodes (concatenate or dot product) → classify.

**Graph-level:** Single output for the whole graph (molecular property prediction, graph classification). Output: pool all node embeddings to a single vector → classify.

---

### Q9: Explain TransE and its limitations. What does RotatE fix?

**TransE:** Models relation r as a translation: h + r ≈ t. Works for simple one-to-one relations.

**Limitations:**
- 1-to-N relations force multiple tail entities to the same embedding
- Symmetric relations force r = 0

**RotatE:** Models relations as complex rotations: t = h ⊙ r with |r\_i| = 1. By working in complex space with unit-modulus constraints, RotatE can model symmetry (rotation by π), antisymmetry (non-π rotation), inversion (conjugate relation), and composition (sequential rotation).

---

### Q10: What happens when GCN runs on a power-law degree distribution?

The symmetric normalization D^{-1/2} A D^{-1/2} weights the contribution of neighbor u to node v by 1/√(deg(u)·deg(v)).

**Consequence:** High-degree hub nodes contribute very little to any individual neighbor (their contribution is divided by their high degree). But hubs receive aggregated contributions from thousands of low-degree neighbors, making their embeddings a heavily averaged representation of a huge neighborhood — information-lossy.

**Over many layers:** Over-smoothing concentrates around high-degree nodes first, because the random walk mixes most quickly through hubs.

**Remedies:** Degree-based feature normalization, capping the number of neighbors (GraphSAGE), or attention mechanisms (GAT) that can learn to ignore uninformative high-degree neighbors.

---

### Q11: How does a GNN handle dynamic graphs?

Standard GNNs are designed for static graphs. Options for dynamic graphs:

1. **Snapshot-based:** Discretize time into snapshots. Run a GNN on each snapshot; propagate the learned embeddings across time with a temporal model (GRU, Transformer).

2. **Temporal Graph Networks (TGN):** Maintain a memory state per node, updated via a GRU when an event occurs involving that node. The GNN aggregation uses both current features and the node's memory state.

3. **Causal masking:** At inference time for node v, only use edges that existed before the prediction timestamp — no future leakage.

---

### Q12: How would you debug a GNN that is not training well?

1. **Check for over-smoothing:** Are node embeddings collapsing? Compute pairwise distances — if near-zero, reduce layers.
2. **Check graph connectivity:** Disconnected components means nodes in different components can never exchange information.
3. **Verify adjacency normalization:** Confirm self-loops were added and symmetric normalization was applied.
4. **Start simple:** Try 1–2 GCN layers before adding complexity. Verify training loss decreases.
5. **Check for data leakage:** In link prediction, ensure train/val/test splits do not share edges in a way that leaks labels.
6. **Baseline comparison:** Does an MLP on node features (ignoring graph structure) already perform well? If so, the graph structure may not be informative, or the GNN is not using it correctly.
7. **Check gradient flow:** Use gradient norms to verify backprop reaches early layers.

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
