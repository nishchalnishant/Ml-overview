# Dynamic Graphs, Temporal GNNs, and Graph Generation

Graphs that change over time and methods to model, predict, and generate graph-structured data.

---

## 1. Dynamic Graphs Overview

A **dynamic graph** captures a system whose nodes, edges, or features evolve. Two fundamental representations exist:

| Paradigm | Representation | Clock | Typical data |
|---|---|---|---|
| Discrete-Time Dynamic Graph (DTDG) | Sequence of snapshots G_1, ..., G_T | Fixed interval | Citation networks, traffic grids |
| Continuous-Time Dynamic Graph (CTDG) | Ordered event stream (u, v, t, feat) | Real-valued timestamp | Social interactions, financial transactions |

**Applications**

- Social networks — friend/follow links appear and disappear
- Financial transactions — detect fraud by modeling interaction bursts
- Traffic — road-segment speeds change every minute
- Molecular dynamics — atom bonds break and form during simulation
- Knowledge graphs — facts have validity intervals

**Core challenges**

- Temporal dependency: future must not leak into past
- Node/edge churn: new entities appear, old ones go dormant
- Long-range temporal patterns coexist with local structural patterns
- Evaluation must respect time ordering (no random splits)

---

## 2. Discrete-Time Dynamic Graphs (DTDG)

### Formulation

A DTDG is a sequence of graph snapshots:

```
G = {G_1, G_2, ..., G_T}   where G_t = (V_t, E_t, X_t)
```

The standard recipe is:

1. Apply a GNN to each snapshot to get node embeddings H_t
2. Feed the sequence H_1, ..., H_T into a temporal model (RNN or Transformer)
3. Use the output for downstream tasks

### EvolveGCN

Instead of fixing GCN weights and using the hidden state to carry memory, EvolveGCN evolves the **weight matrices** of the GCN itself using an RNN. This is particularly useful when the node set changes across snapshots.

```
W_t = GRU(W_{t-1}, H_{t-1})    # weights are the RNN hidden state
H_t = GCN(A_t, X_t; W_t)
```

Two variants:
- **EvolveGCN-H**: uses node embedding matrix H_{t-1} as input to GRU
- **EvolveGCN-O**: uses the previous weight matrix W_{t-1} directly (no node features needed)

Advantage: handles node appearance/disappearance gracefully because the weight matrices are decoupled from node identity.

### GCRN (Graph Convolutional RNN)

GCRN replaces the linear transformation inside an LSTM/GRU cell with a graph convolution:

```
# Standard GRU gate:
z_t = sigmoid(W_z x_t + U_z h_{t-1})

# GCRN gate — W_z replaced by spectral/spatial graph conv:
z_t = sigmoid(GraphConv(A_t, x_t) + GraphConv(A_t, h_{t-1}))
```

The graph convolution propagates spatial information while the gating propagates temporal information within a single unified cell.

### DTDG Training Tips

- Use the same GNN architecture per snapshot (weight sharing reduces parameters)
- Truncated BPTT through time to manage memory on long sequences
- Snapshot graph may need padding if V_t differs across t
- Time gap between snapshots should be uniform or explicitly encoded

---

## 3. Continuous-Time Dynamic Graphs (CTDG)

### Formulation

A CTDG is an ordered stream of timestamped events:

```
E = {(u_1, v_1, t_1, f_1), (u_2, v_2, t_2, f_2), ...}   t_1 <= t_2 <= ...
```

To embed node u at query time t, we consider its **temporal neighborhood**:

```
N(u, t) = {(v, t', f) : (u, v, t') in E, t' < t}
```

### TGAT (Temporal Graph Attention)

TGAT encodes time directly into the attention mechanism using **time encoding** derived from Bochner's theorem: any positive-definite kernel k(t1, t2) can be expressed as the expectation of a random feature map over a spectral distribution.

**Time encoding via random Fourier features:**

```python
class TimeEncoding(nn.Module):
    def __init__(self, d_time):
        super().__init__()
        # Learnable frequencies (initialized randomly, fine-tuned)
        self.w = nn.Parameter(torch.randn(d_time // 2))
        self.b = nn.Parameter(torch.randn(d_time // 2))

    def forward(self, t):
        # t: (B,) float tensor of timestamps
        t = t.unsqueeze(-1)             # (B, 1)
        x = t * self.w + self.b         # (B, d_time//2)
        return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)  # (B, d_time)
```

The encoded time is concatenated with edge features before temporal attention is computed over the neighborhood N(u, t).

**TGAT layer:**

```
h_u(t) = Attn(Q=phi(u,t),  K=phi(v, t') for (v,t') in N(u,t),
               V=cat(feat_{uv}, TimeEnc(t - t')))
```

Using relative time `t - t'` instead of absolute timestamps improves generalization.

### TGN (Temporal Graph Networks)

TGN is a general framework that separates memory, message computation, and graph embedding into explicit modules.

**Key components:**

```
Memory:     s_u in R^d  — persistent state per node, updated after each event
Messages:   m_u(t) = msg(s_u(t^-), s_v(t^-), delta_t, e_{uv}(t))
Aggregation: agg_u(t) = AGG({m_u(t') : t' <= t})
Memory update: s_u(t) = mem(agg_u(t), s_u(t^-))   # e.g. GRU
Embedding:  z_u(t) = GNN(s_u(t), N(u,t))           # temporal graph attention
```

**Simplified TGN forward pass sketch:**

```python
class TGN(nn.Module):
    def __init__(self, num_nodes, mem_dim, time_dim, feat_dim):
        super().__init__()
        self.memory = torch.zeros(num_nodes, mem_dim)  # persistent
        self.time_enc = TimeEncoding(time_dim)
        self.msg_fn = nn.Linear(2 * mem_dim + time_dim + feat_dim, mem_dim)
        self.mem_update = nn.GRUCell(mem_dim, mem_dim)
        self.embedding = TemporalGraphAttention(mem_dim, time_dim)

    def compute_messages(self, src, dst, t, edge_feat):
        dt = self.time_enc(t)
        raw = torch.cat([self.memory[src], self.memory[dst], dt, edge_feat], dim=-1)
        return src, self.msg_fn(raw)  # (node_ids, messages)

    def update_memory(self, node_ids, messages):
        self.memory[node_ids] = self.mem_update(messages, self.memory[node_ids])

    def forward(self, src, dst, t, edge_feat, neg_dst=None):
        # 1. Compute messages from raw events
        node_ids, msgs = self.compute_messages(src, dst, t, edge_feat)
        # 2. Update memory (detach to avoid BPTT through entire history)
        self.update_memory(node_ids, msgs.detach())
        # 3. Embed using updated memory + temporal neighborhood
        z_src = self.embedding(src, t, self.memory)
        z_dst = self.embedding(dst, t, self.memory)
        return z_src, z_dst
```

**Memory staleness problem:** A node's memory is only updated when it appears in an event. Rarely-active nodes carry stale memories. TGN addresses this by updating both source and destination memories at each event.

---

## 4. Temporal Link Prediction

### Task Definition

Given all events up to time t, predict whether edge (u, v) will appear at time t' > t.

### Evaluation Protocol

**Time-based splits — mandatory, no random splits:**

```
|--------- train ----------|-- val --|-- test --|
t=0                       t1        t2         t_max
```

- All training edges precede all validation edges which precede all test edges
- Random splits cause temporal leakage (future supervises past)

**Transductive vs Inductive:**

| Setting | Nodes at test time | Challenge |
|---|---|---|
| Transductive | Seen during training | Standard generalization |
| Inductive | New nodes not in training | Model must generalize to unseen nodes |

TGN handles inductive setting because memory is computed from event history, not a lookup table.

### Scoring

```python
# Binary cross-entropy with negative sampling
pos_score = (z_src * z_dst).sum(-1)           # dot product
neg_score = (z_src * z_neg_dst).sum(-1)       # random negative dst
loss = F.binary_cross_entropy_with_logits(
    torch.cat([pos_score, neg_score]),
    torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)])
)
```

**Metrics:** Average Precision (AP), ROC-AUC, Mean Reciprocal Rank (MRR)

**Negative sampling strategy matters:** Random negatives are easy. Historical negatives (edges that existed but no longer do) or hard negatives (structurally similar non-edges) are more realistic.

---

## 5. Graph Generation

### Why Graph Generation is Hard

1. **Variable size** — unlike images, graphs have no fixed n x n grid
2. **Permutation invariance** — the same graph has n! adjacency matrix representations; the model must be invariant or learn to handle this
3. **Validity constraints** — chemical graphs require valid valencies, connectivity, no isolated atoms
4. **Sparsity** — adjacency matrices are sparse; naively modeling all n^2 entries is wasteful and penalizes sparsity poorly
5. **Evaluation** — no pixel-wise MSE; need domain-specific metrics (chemical validity, graph statistics)

---

## 6. VGAE — Variational Graph Autoencoder

### Architecture

VGAE (Kipf & Welling, 2016) applies the VAE framework to graphs. It encodes node features via GCN into a latent space, then decodes by taking the inner product of latent vectors to reconstruct the adjacency matrix.

```
Encoder:  Z = GCN(A, X)   ->  mu, log_sigma  (reparameterize to z_i in R^d)
Decoder:  A_hat_{ij} = sigmoid(z_i^T z_j)
```

**ELBO objective:**

```
L = E_q[log p(A|Z)] - KL(q(Z|A,X) || p(Z))
```

The reconstruction term maximizes the likelihood of observed edges; the KL term regularizes the latent space toward a standard Gaussian.

### PyG Implementation

```python
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import negative_sampling, to_dense_adj


class GCNEncoder(torch.nn.Module):
    """Two-layer GCN encoder producing mean and log-variance."""

    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv_mu = GCNConv(hidden_channels, out_channels)
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
        z = self.reparameterize(self.mu, self.logstd)
        return z

    def decode(self, z, edge_index):
        """Inner-product decoder for a given set of edges."""
        src, dst = edge_index
        return (z[src] * z[dst]).sum(dim=-1)

    def decode_all(self, z):
        """Full n x n probability matrix (use only for small graphs)."""
        return torch.sigmoid(z @ z.t())

    def kl_loss(self):
        return -0.5 * torch.mean(
            torch.sum(1 + 2 * self.logstd - self.mu.pow(2) - self.logstd.exp().pow(2), dim=1)
        )

    def recon_loss(self, z, pos_edge_index, neg_edge_index=None):
        """Binary cross-entropy reconstruction loss with negative sampling."""
        pos_loss = -torch.log(torch.sigmoid(self.decode(z, pos_edge_index)) + 1e-15).mean()

        if neg_edge_index is None:
            neg_edge_index = negative_sampling(
                pos_edge_index,
                num_nodes=z.size(0),
                num_neg_samples=pos_edge_index.size(1),
            )
        neg_loss = -torch.log(1 - torch.sigmoid(self.decode(z, neg_edge_index)) + 1e-15).mean()
        return pos_loss + neg_loss

    def forward(self, x, edge_index):
        z = self.encode(x, edge_index)
        return z


# --- Training loop ---

def train_vgae(data, epochs=200, latent_dim=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)

    encoder = GCNEncoder(
        in_channels=data.num_features,
        hidden_channels=32,
        out_channels=latent_dim,
    )
    model = VGAE(encoder).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        z = model.encode(data.x, data.edge_index)
        recon = model.recon_loss(z, data.edge_index)
        kl = model.kl_loss()
        loss = recon + (1 / data.num_nodes) * kl  # beta-weighted KL

        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Recon: {recon:.4f} | KL: {kl:.4f}")

    return model


# --- Sampling new graphs ---

def sample_graph(model, z_dim=16, n_nodes=20, threshold=0.5):
    """Sample a new graph by drawing z from prior and decoding."""
    model.eval()
    with torch.no_grad():
        z = torch.randn(n_nodes, z_dim)
        adj = model.decode_all(z)          # (n, n) probability matrix
        adj = (adj > threshold).float()
        adj = adj.triu(diagonal=1)         # upper triangle -> undirected
        adj = adj + adj.t()                # symmetrize
    return adj
```

### VGAE Limitations

- Decoding requires O(n^2) inner products — does not scale to large graphs
- No validity guarantee on generated graphs
- Latent space models nodes independently (no global graph-level variable)

---

## 7. Sequential Graph Generation

### GraphRNN

GraphRNN (You et al., 2018) generates graphs one node at a time. At each step it adds a new node and decides which existing nodes it connects to.

**Process:**

```
For each new node v_i:
    1. Graph-level RNN updates hidden state: h_i = f_graph(h_{i-1})
    2. Edge-level RNN generates a binary sequence e_{i,1}, e_{i,2}, ..., e_{i,i-1}
       where e_{i,j} = 1 means edge (v_i, v_j) exists
    3. Stop when EOS token is generated
```

**Key insight:** BFS node ordering dramatically reduces the sequence length — for sparse graphs, only the last few nodes in BFS order can connect to the new node (bandwidth of the adjacency matrix is small). This turns an O(n^2) edge sequence into O(n * M) where M is the BFS bandwidth.

**Limitation:** Sequential generation is slow at inference; graphs of n=1000 nodes require n^2 RNN steps in the worst case.

### GRAN (Graph Recurrent Attention Network)

GRAN generates a **block** of nodes at each step instead of one node, using graph attention over the partially generated graph:

```
At step t:
    - Candidate block B_t = {v_{k+1}, ..., v_{k+b}}
    - Run GAT over current graph + candidates to score edges
    - Sample edges between B_t and existing nodes
    - Add B_t to graph, repeat
```

Block generation + attention allows GRAN to model long-range dependencies within a generation step, achieving better quality than GraphRNN on large graphs while being parallelizable within blocks.

---

## 8. Diffusion-Based Graph Generation

### GDSS (Graph Diffusion Score-based model, 2022)

GDSS jointly diffuses the node feature matrix X and adjacency matrix A:

```
Forward:  (X_T, A_T) ~ N(0, I) x N(0, I)   (Gaussian noise)
Reverse:  score model s_theta learns grad_{X,A} log p_t(X, A)
          jointly denoises both X and A at each reverse step
```

The score function must be permutation-equivariant. GDSS uses a GNN to compute scores over the current noisy (X_t, A_t).

### DiGress (2023)

DiGress works in **discrete space**: the diffusion process adds/removes edges and changes node types according to a categorical (Markov) noise process, rather than adding Gaussian noise.

```
Forward:  corrupt graph by randomly adding/removing edges with probability beta_t
Reverse:  transformer predicts clean graph G_0 from noisy G_t (denoising approach)
```

DiGress naturally handles discrete node/edge attributes and produces sparser, more valid graphs than continuous diffusion approaches, especially for molecular graphs.

---

## 9. Molecule Generation

### Representation

| Format | Example | Pros | Cons |
|---|---|---|---|
| SMILES string | `CC(=O)Oc1ccccc1C(=O)O` | Compact, standard | Invalid strings common, not unique |
| Graph | Nodes=atoms, Edges=bonds | Natural, validity-checkable | Permutation invariance |
| 3D point cloud | Atom xyz coordinates | Captures geometry | Loses bonding topology |

### Validity Constraints

A generated molecular graph is **chemically valid** if:
- Each atom satisfies its valency (Carbon: degree <= 4, Nitrogen <= 3, etc.)
- The graph is connected (no isolated fragments unless explicitly modeled)
- No invalid bond types (no triple bonds between atoms that cannot form them)

These constraints are hard to enforce during generation. Common approaches:
- Post-hoc validity filtering (discard invalid samples)
- Valency-aware decoding (mask invalid bond actions)
- Junction Tree decomposition

### Junction Tree VAE (JT-VAE)

JT-VAE (Jin et al., 2018) decomposes molecules into **ring systems and functional groups** (motifs from a fixed vocabulary), builds a junction tree over these motifs, and generates:

1. The junction tree structure (which motifs and how they connect)
2. The assembly of motifs into a full molecule (resolve attachment points)

Because motifs are valid by construction and the assembly respects valency rules, JT-VAE achieves near-100% validity. The trade-off is that the vocabulary must be pre-computed and limits novelty to combinations of known motifs.

---

## 10. Evaluation Metrics

### Graph Generation (General)

| Metric | Description |
|---|---|
| **Validity** | Fraction of generated graphs satisfying domain constraints (e.g., connectivity) |
| **Uniqueness** | Fraction of valid generated graphs that are non-duplicate |
| **Novelty** | Fraction of valid unique graphs not present in the training set |
| **Degree distribution** | MMD between degree histograms of generated vs real graphs |
| **Clustering coefficient** | MMD on local clustering distributions |
| **Orbit statistics** | MMD on counts of 4-node subgraph patterns (captures higher-order structure) |

**MMD (Maximum Mean Discrepancy)** is the standard distance metric for comparing distributions of graph statistics.

### Molecule Generation

| Metric | Description |
|---|---|
| **Validity** | Fraction passing RDKit sanitization |
| **Uniqueness** | Fraction unique among valid samples |
| **Novelty** | Fraction not in training set |
| **FCD** | Fréchet ChemNet Distance — like FID for images; uses penultimate layer of a pretrained ChemNet on SMILES; lower is better |
| **SA Score** | Synthetic accessibility score (1=easy, 10=hard); lower generated SA = more synthesizable |
| **QED** | Quantitative estimate of drug-likeness (0–1); Lipinski rules encoded |
| **MOSES / GuacaMol benchmarks** | Standardized test suites for molecular generation |

### FCD Computation

```python
# Conceptual sketch — uses fcd library
from fcd import get_fcd, load_ref_model, canonical_smiles

ref_model = load_ref_model()
ref_smiles = [...]      # training set
gen_smiles = [...]      # generated molecules (valid only)

ref_smiles_can = canonical_smiles(ref_smiles)
gen_smiles_can = canonical_smiles(gen_smiles)

fcd_score = get_fcd(gen_smiles_can, ref_smiles_can, ref_model)
```

FCD captures both chemical structure and bioactivity distribution; a model with high validity/uniqueness but wrong chemical distribution will show high FCD.

---

## 11. Key Interview Points

**Dynamic graphs:**

- Know the DTDG vs CTDG distinction and when each applies. CTDG is more expressive but more complex to implement.
- EvolveGCN's core idea is evolving GCN *weights* with an RNN — the weight matrix IS the hidden state. This decouples temporal modeling from node identity.
- TGN's memory module gives nodes persistent state between interactions; without memory, a node seen once can only be embedded from its current neighborhood.
- TGAT time encoding uses random Fourier features to map scalar timestamps to vectors that are compatible with attention. The key property: the kernel between two times is a function only of their difference.
- Temporal link prediction must use time-based splits. Random splits cause leakage.

**Graph generation:**

- Permutation invariance is the central difficulty. GraphRNN sidesteps it with a canonical ordering (BFS); diffusion models handle it via equivariant score networks.
- VGAE decodes via inner product — latent geometry directly encodes edge probability. Close nodes in latent space get high edge probability.
- The KL term in VGAE scales as 1/n to prevent it from dominating when n is large.
- GraphRNN is autoregressive and exact but slow. Diffusion models are faster to sample (parallel denoising) but require approximation.
- JT-VAE achieves near-100% validity by generating motifs from a valid vocabulary rather than individual atoms.
- Validity alone is insufficient — a model that always generates methane (CH4) has 100% validity but zero novelty and usefulness. Report all three: validity, uniqueness, novelty.
- FCD is the molecule generation analog of FID: it measures whether the distribution of generated molecules (in a learned embedding space) matches the training distribution, not just individual sample quality.
- For interviews: always distinguish between *evaluation* metrics (validity, FCD) and *optimization* objectives (ELBO, score matching loss) — models are not directly trained to maximize validity.
