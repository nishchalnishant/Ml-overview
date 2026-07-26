---
module: Architectures
topic: Architectures
subtopic: Graph Neural Networks
status: unread
tags: [gnn, gcn, graphsage, gat, message-passing, deeplearning]
prerequisites: [backpropagation, linear-algebra, attention]
---
# Graph Neural Networks

---

## TL;DR

GNNs generalize convolution to irregular, non-Euclidean structure: instead of a fixed spatial neighborhood (CNN) or a fixed sequence order (RNN/Transformer), each node aggregates information from its graph neighbors. Nearly every variant (GCN, GraphSAGE, GAT, GIN) is an instance of **message passing**: compute messages from neighbors, aggregate them permutation-invariantly, update the node's representation. The central failure mode to know cold is **over-smoothing** — stack too many layers and every node's representation converges to the same point.

---

## The problem

CNNs assume a regular grid (pixels have a fixed number of neighbors in fixed relative positions). RNNs and Transformers assume a sequence (each token has a well-defined predecessor/successor, or at least a full ordered set to attend over). Graphs have neither: a node can have any number of neighbors, in no canonical order — a social network, a molecule, a citation graph, a knowledge graph. Applying a CNN or RNN to a graph requires imposing an artificial ordering, which throws away the actual structure and produces outputs that aren't invariant to relabeling nodes. A GNN's core requirement is **permutation invariance**: the output shouldn't change if you relabel/reorder the nodes, only if the underlying graph structure changes.

## The core insight: message passing

Every mainstream GNN layer follows the same three-step template, for node $v$ with neighbors $\mathcal{N}(v)$:

1. **Message**: compute a message from each neighbor $u \in \mathcal{N}(v)$, typically a function of $u$'s current representation (and sometimes the edge features).
2. **Aggregate**: combine incoming messages with a permutation-invariant operator — sum, mean, or max. This is what guarantees the layer doesn't depend on neighbor ordering.
3. **Update**: combine the aggregated message with the node's own previous representation to produce the new representation.

$$h_v^{(k)} = \text{UPDATE}\Big(h_v^{(k-1)},\ \text{AGGREGATE}\big(\{h_u^{(k-1)} : u \in \mathcal{N}(v)\}\big)\Big)$$

Stacking $k$ layers means node $v$'s final representation has aggregated information from every node within $k$ hops — the receptive field grows with depth, exactly like a CNN's receptive field grows with conv layers.

---

## GCN (Graph Convolutional Network)

**The mechanics**: a GCN layer is a symmetric-normalized aggregation followed by a linear transform and nonlinearity:

$$H^{(k)} = \sigma\left(\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} H^{(k-1)} W^{(k)}\right)$$

- $\tilde{A} = A + I$: adjacency matrix with self-loops added (a node includes itself in its own aggregation)
- $\tilde{D}$: degree matrix of $\tilde{A}$
- $\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2}$: symmetric normalization — without it, high-degree nodes would dominate purely because they sum more terms, not because their neighbors are more informative

This is literally the message-passing template with mean-ish (degree-normalized) aggregation and a shared linear transform as the update. **Every node uses the same weight matrix $W^{(k)}$** — this is the graph analogue of weight sharing in CNNs.

**What breaks**: GCN aggregation is fixed (not learned) and transductive in its original form — it operates on one fixed graph, so it can't naturally generalize to unseen nodes at inference time.

## GraphSAGE (inductive generalization)

**The problem GCN doesn't solve**: production graphs grow (new users, new items) — you need to embed a node you've never seen during training, without retraining on the whole graph.

**The fix**: instead of a fixed normalized adjacency, GraphSAGE learns an aggregation *function* and applies it to a **sampled** fixed-size subset of neighbors, not the full neighborhood:

$$h_v^{(k)} = \sigma\left(W^{(k)} \cdot \text{CONCAT}\big(h_v^{(k-1)},\ \text{AGG}(\{h_u^{(k-1)} : u \in \mathcal{S}(v)\})\big)\right)$$

where $\mathcal{S}(v)$ is a fixed-size random sample of $v$'s neighbors (not all of them). Because the aggregation function itself is learned and shared (not tied to a specific graph's fixed adjacency matrix), it generalizes to nodes/graphs not seen during training — this is what "inductive" means here. Neighbor sampling also bounds the per-node compute cost, which matters at scale (a node with 10,000 neighbors doesn't blow up the batch).

## GAT (Graph Attention Network)

**The problem GCN and GraphSAGE don't solve**: both treat all neighbors as equally (or degree-) important. Not all edges carry equal information — in a citation graph, some cited papers are far more relevant to a node's topic than others.

**The fix**: learn an attention weight per edge, same idea as Transformer attention but restricted to graph neighbors instead of all tokens:

$$\alpha_{vu} = \frac{\exp\big(\text{LeakyReLU}(a^T [W h_v \| W h_u])\big)}{\sum_{u' \in \mathcal{N}(v)} \exp\big(\text{LeakyReLU}(a^T [W h_v \| W h_{u'}])\big)}$$

$$h_v^{(k)} = \sigma\left(\sum_{u \in \mathcal{N}(v)} \alpha_{vu} W h_u^{(k-1)}\right)$$

Multi-head attention (as in Transformers) stabilizes training and lets different heads attend to different relational patterns. GAT is strictly more expressive than GCN's fixed normalization, at the cost of more parameters and compute per layer.

## GIN (Graph Isomorphism Network) — expressiveness ceiling

**Why this matters**: GCN/GraphSAGE-style mean/max aggregation cannot distinguish some non-isomorphic graphs — mean and max aggregation are provably weaker than sum aggregation at telling multisets of neighbor features apart (two different neighbor multisets can produce the same mean). GIN uses sum aggregation plus an MLP and is proven to be as powerful as the **Weisfeiler-Lehman (WL) graph isomorphism test**, the theoretical upper bound for this class of message-passing GNN:

$$h_v^{(k)} = \text{MLP}^{(k)}\left((1+\epsilon) \cdot h_v^{(k-1)} + \sum_{u \in \mathcal{N}(v)} h_u^{(k-1)}\right)$$

Know this fact even if you never implement GIN: **sum > max > mean** in expressive power for aggregation, and no message-passing GNN can exceed 1-WL expressiveness without additional structural features.

---

## Common traps

**Trap: stacking many GNN layers to increase receptive field, like a deep CNN.**
Deep CNNs work because pixels stay distinguishable across layers. GNNs suffer **over-smoothing**: after enough message-passing rounds, every node's representation converges toward the graph average (or the average within its connected component), because repeated averaging is a contraction. Past ~2-4 layers on most graphs, accuracy degrades rather than improves. This is the single most-tested GNN failure mode — an interviewer asking "why not just make it deeper" is testing whether you know this.

**Trap: forgetting self-loops or the residual/update step.**
Without a self-loop ($\tilde{A} = A+I$) or a residual-style update term, a node's own information gets diluted or lost entirely by aggregation, since $h_v^{(k)}$ becomes purely a function of neighbors.

**Trap: assuming inference is free once trained (like a CNN on a new image).**
For a transductive GCN, a genuinely new node requires access to its neighbors' representations and effectively a forward pass over (part of) the graph — there's no fixed-size feature vector to just feed through independently. This is why GraphSAGE's neighbor-sampling inductive design matters operationally, not just theoretically.

**Trap: treating "aggregate" as always mean.**
Interviewers probe whether you know the expressiveness ordering (sum > max > mean) and can justify why a specific aggregator was chosen for a specific task — e.g., sum for tasks sensitive to neighbor *count* (molecule property prediction), mean when neighbor count varies wildly and you want scale invariance.

---

## Interview Angles

**Q1: Derive why GCN needs the symmetric normalization $\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}$ instead of just $\tilde{A}$.**

Without normalization, a node's aggregated signal scales with its degree — high-degree nodes produce systematically larger activations purely from summing more terms, not because they're more informative. Row-normalizing ($\tilde{D}^{-1}\tilde{A}$) fixes the scale but is asymmetric — it becomes a random-walk transition matrix, not a symmetric operator, which complicates the spectral-graph-theory interpretation GCN is derived from (a first-order approximation of spectral graph convolution). Symmetric normalization keeps the operator symmetric (real eigenvalues, more stable optimization) while still controlling for degree.

**Q2: Why is over-smoothing worse in GNNs than vanishing gradients are in deep RNNs, conceptually?**

Vanishing gradients are an optimization problem — the *representations themselves* are still distinct, but the gradient signal to update them decays. Over-smoothing is a representational collapse — the forward-pass features actually become indistinguishable across nodes as depth increases, independent of how well you can optimize. More/better optimization doesn't fix it; you need architectural mitigations (residual connections, fewer layers, jumping knowledge/skip aggregation across layers, or explicit anti-smoothing regularization).

**Q3: You have a graph with 50M nodes. Full-batch GCN training (multiplying the full adjacency matrix every forward pass) doesn't fit in memory. What do you do?**

Neighbor sampling (GraphSAGE-style): sample a fixed-size subset of neighbors per node per layer, and only construct the computational subgraph needed for the current mini-batch of target nodes. This bounds both memory and the "neighbor explosion" problem where a $k$-layer GNN's receptive field can include an exponentially growing number of nodes for high-degree graphs. Cluster-GCN (partition the graph, train on subgraphs) is the alternative when even sampled neighborhoods are too large.

**Q4: When would you *not* reach for a GNN, even though your data is technically a graph?**

When the graph structure isn't actually predictive — e.g., a k-NN graph built purely as a preprocessing convenience over tabular data, where a plain MLP or GBT on the raw features performs just as well and is far cheaper to train/serve. Also when the graph is small and dense enough that it's really just a fully-connected relational structure — attention over all nodes (a small Transformer) may be simpler and equally expressive. GNNs earn their complexity when the graph topology itself encodes information the model needs (molecule bonds, social network ties, citation structure).

---

## Where to Next

- **Attention mechanics that GAT borrows from** → [02-attention.md](02-attention.md)
- **Where SSM/Mamba (the other "beyond fixed-grid" architecture family) actually lives** → [01-rnn-lstm-gru.md](01-rnn-lstm-gru.md)
- **Architecture selection framework (when to reach for which model family)** → [04-dl-architectures.md](04-dl-architectures.md)
- **Recommender systems / graph-structured applied domains** → [07-domains/](../07-domains/)
