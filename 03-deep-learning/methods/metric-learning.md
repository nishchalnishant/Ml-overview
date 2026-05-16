# Metric Learning — Comprehensive Reference

Learn an embedding space where similar items are geometrically close and dissimilar items are far apart, enabling similarity search, verification, and few-shot recognition without retraining a classifier.

---

## Table of Contents

1. What is Metric Learning
2. Contrastive Loss
3. Triplet Loss and Semi-Hard Negative Mining
4. N-pair / Multi-class N-pair Loss
5. ArcFace / CosFace / SphereFace
6. Supervised Contrastive Learning (SupCon)
7. SimCLR Revisited for Retrieval
8. Image Retrieval Pipeline with FAISS
9. Evaluation Metrics
10. Person Re-Identification (Re-ID)
11. Key Interview Points

---

## 1. What is Metric Learning

Metric learning trains a function `f: X -> R^d` such that a chosen distance (usually L2 or cosine) reflects semantic similarity:

```
d(f(x_i), f(x_j)) small   if x_i, x_j are same class
d(f(x_i), f(x_j)) large   if x_i, x_j are different classes
```

The embedding space is **not** tied to a fixed set of classes — the model generalises to unseen identities at inference time, unlike softmax classifiers.

### Core Applications

| Domain | Task |
|---|---|
| Face verification | Is this the same person? (1:1 match) |
| Face identification | Who is this? (1:N search) |
| Image retrieval | Find visually similar images in a gallery |
| Few-shot learning | Classify new classes from 1–5 labelled examples |
| Product search | Find same/similar SKUs from a photo |
| Person Re-ID | Match pedestrians across cameras |

### Design choices

- **Backbone**: ResNet, ViT, EfficientNet
- **Embedding head**: global average pool → FC → L2-normalise
- **Loss**: contrastive, triplet, margin, angular (ArcFace)
- **Mining**: offline vs. online hard/semi-hard negative mining

---

## 2. Contrastive Loss

Proposed by Hadsell et al. (2006). Operates on **pairs** `(x_i, x_j)` with a binary label `y = 1` (same class) or `y = 0` (different class).

### Formula

```
L = y * d^2 + (1 - y) * max(0, margin - d)^2

where d = ||f(x_i) - f(x_j)||_2
```

- Positive pairs (`y=1`): loss pulls embeddings together.
- Negative pairs (`y=0`): loss pushes embeddings apart only when distance < margin. Pairs already beyond the margin contribute zero loss.

### Weaknesses

- Requires curated positive/negative pairs — quadratic in dataset size.
- Sensitive to margin hyperparameter.
- Collapsed negatives: most random pairs are already easy and contribute nothing.

### Code

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, emb_a: torch.Tensor, emb_b: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        emb_a, emb_b : (B, D) L2-normalised embeddings
        labels        : (B,)  1 = same class, 0 = different class
        """
        dist = F.pairwise_distance(emb_a, emb_b, p=2)          # (B,)
        pos_loss = labels * dist.pow(2)
        neg_loss = (1 - labels) * F.relu(self.margin - dist).pow(2)
        return (pos_loss + neg_loss).mean()


# --- Pairs dataset skeleton ---
class PairsDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
        # Pre-generate pairs (or sample on the fly)
        self.pairs = self._build_pairs()

    def _build_pairs(self):
        from itertools import combinations
        import random
        pairs = []
        class_to_idx = {}
        for i, lbl in enumerate(self.labels):
            class_to_idx.setdefault(lbl, []).append(i)
        # Positive pairs
        for lbl, idxs in class_to_idx.items():
            for a, b in combinations(idxs, 2):
                pairs.append((a, b, 1))
        # Negative pairs (sample to balance)
        classes = list(class_to_idx.keys())
        for _ in range(len(pairs)):
            ca, cb = random.sample(classes, 2)
            a = random.choice(class_to_idx[ca])
            b = random.choice(class_to_idx[cb])
            pairs.append((a, b, 0))
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        ia, ib, label = self.pairs[idx]
        xa = self.images[ia]
        xb = self.images[ib]
        if self.transform:
            xa = self.transform(xa)
            xb = self.transform(xb)
        return xa, xb, torch.tensor(label, dtype=torch.float32)
```

---

## 3. Triplet Loss and Semi-Hard Negative Mining

Proposed by Schroff et al. (FaceNet, 2015). Operates on **triplets** `(a, p, n)`:
- **Anchor** `a`: reference sample
- **Positive** `p`: same class as anchor
- **Negative** `n`: different class from anchor

### Formula

```
L = max(0, d(a, p) - d(a, n) + margin)

where d = ||f(x) - f(y)||_2
```

The loss is zero when negatives are already `margin` farther than positives.

### Why Mining Matters

With random triplets most negatives are **easy** (already far away) and contribute zero gradient. Hard negatives (closer than positives) cause **collapsed** embeddings. Semi-hard negatives sit in the sweet spot:

```
d(a, p) < d(a, n) < d(a, p) + margin
```

They violate the margin but are not the hardest — empirically the most stable training signal.

### Online Mining Code

```python
import torch
import torch.nn.functional as F


def pairwise_distances(embeddings: torch.Tensor) -> torch.Tensor:
    """Compute all-pairs squared L2 distances. (B, B)"""
    dot = embeddings @ embeddings.T                         # (B, B)
    sq  = dot.diagonal().unsqueeze(1)                       # (B, 1)
    dist2 = sq + sq.T - 2 * dot                            # (B, B)
    return F.relu(dist2).sqrt()                            # numerical safety


def online_triplet_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 0.3,
    mining: str = "semihard",   # "semihard" | "hard" | "all"
) -> torch.Tensor:
    """
    embeddings : (B, D) L2-normalised
    labels     : (B,)  integer class ids
    """
    dist = pairwise_distances(embeddings)                   # (B, B)
    B = embeddings.size(0)

    # Masks
    labels_eq  = labels.unsqueeze(0) == labels.unsqueeze(1)  # (B, B)
    labels_neq = ~labels_eq
    not_self   = ~torch.eye(B, dtype=torch.bool, device=embeddings.device)

    pos_mask = labels_eq & not_self     # valid positives
    neg_mask = labels_neq               # valid negatives

    # For each anchor pick hardest positive
    d_ap = (dist * pos_mask.float()).max(dim=1).values      # (B,)

    loss_vals = []
    for i in range(B):
        d_pos = d_ap[i]
        neg_dists = dist[i][neg_mask[i]]                    # distances to all negatives

        if mining == "semihard":
            # negatives beyond d_pos but within d_pos + margin
            mask = (neg_dists > d_pos) & (neg_dists < d_pos + margin)
            candidates = neg_dists[mask]
            if candidates.numel() == 0:
                candidates = neg_dists   # fall back to all negatives
        elif mining == "hard":
            candidates = neg_dists
        else:  # all
            candidates = neg_dists

        d_neg = candidates.min()
        loss_vals.append(F.relu(d_pos - d_neg + margin))

    return torch.stack(loss_vals).mean()


# Training loop sketch
# model    : backbone + embedding head with L2 norm
# loader   : yields (images, labels) batches with many classes per batch

def train_step(model, optimizer, images, labels, margin=0.3):
    optimizer.zero_grad()
    emb = model(images)                                     # (B, D), L2-normalised
    loss = online_triplet_loss(emb, labels, margin=margin, mining="semihard")
    loss.backward()
    optimizer.step()
    return loss.item()
```

### Training Tips

- Use **large batches** (512+) with many classes per batch to find informative triplets.
- Start with semi-hard mining; switch to hard mining after warm-up.
- Monitor `fraction_positive_triplets` — if it drops near zero, training has converged or collapsed.
- L2-normalise embeddings before computing distances.

---

## 4. N-pair / Multi-class N-pair Loss

Kang et al. (2016). Generalises contrastive loss to use **all negatives in the batch simultaneously**, avoiding repeated pair construction.

### Setup

Sample N classes, 2 examples each: `{(a_i, p_i)}_{i=1}^N`.
For anchor `a_i`, positive is `p_i`, negatives are all other `{p_j}_{j≠i}`.

### Formula

```
L = (1/N) * sum_i log(1 + sum_{j≠i} exp(a_i^T p_j - a_i^T p_i))
```

This is equivalent to softmax cross-entropy where the logits are inner products.

### Relationship to NT-Xent (SimCLR)

NT-Xent (Normalized Temperature-scaled Cross-Entropy) is essentially N-pair loss with temperature scaling:

```
L_i = -log [ exp(sim(a_i, p_i)/tau) / sum_{j≠i} exp(sim(a_i, x_j)/tau) ]
```

### Code

```python
import torch
import torch.nn.functional as F


def npair_loss(anchors: torch.Tensor, positives: torch.Tensor, l2_reg: float = 0.002) -> torch.Tensor:
    """
    anchors   : (N, D) L2-normalised anchor embeddings
    positives : (N, D) L2-normalised positive embeddings
    Each anchor's positive is the corresponding row; all others are negatives.
    """
    # Similarity matrix: (N, N)  — row i vs all positives
    logits = anchors @ positives.T                          # (N, N)
    # Targets: diagonal (anchor i matches positive i)
    targets = torch.arange(logits.size(0), device=logits.device)
    loss = F.cross_entropy(logits, targets)
    # Optional L2 regularisation on embedding norms
    reg = l2_reg * (anchors.norm(dim=1).pow(2).mean() + positives.norm(dim=1).pow(2).mean())
    return loss + reg
```

---

## 5. ArcFace / CosFace / SphereFace

Angular margin losses treat the final classification layer weights as **class prototypes** on a hypersphere and inject a margin in angular space. They produce highly discriminative embeddings without special mining.

### Family Overview

| Method | Margin type | Decision boundary |
|---|---|---|
| SphereFace (Liu 2017) | Multiplicative angular | `cos(m*theta_yi) > cos(theta_j)` |
| CosFace (Wang 2018) | Additive cosine | `cos(theta_yi) - m > cos(theta_j)` |
| ArcFace (Deng 2019) | Additive angular | `cos(theta_yi + m) > cos(theta_j)` |

ArcFace is the most popular — the additive angular margin has a clear geometric interpretation on the unit hypersphere.

### ArcFace Formula

```
L = -log [ exp(s * cos(theta_yi + m)) / (exp(s * cos(theta_yi + m)) + sum_{j≠yi} exp(s * cos(theta_j))) ]

theta_yi = arccos(W_yi^T * f / (||W_yi|| * ||f||))
```

- `s`: scale (feature scale, typically 64)
- `m`: angular margin (typically 0.5 radians ~ 28.6 degrees)
- `W`: learnable class prototype matrix, L2-normalised each forward pass
- `f`: L2-normalised embedding

### ArcFace Layer Code

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceHead(nn.Module):
    """
    Drop-in classification head that applies additive angular margin.
    Use in place of a standard nn.Linear + CrossEntropyLoss.
    """

    def __init__(self, in_features: int, num_classes: int, s: float = 64.0, m: float = 0.5):
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

        # Pre-compute margin values for numerical stability
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th    = math.cos(math.pi - m)  # threshold: cos(pi - m)
        self.mm    = math.sin(math.pi - m) * m

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        features : (B, D) raw embeddings (NOT yet normalised)
        labels   : (B,)  integer class ids in [0, num_classes)
        returns  : scalar cross-entropy loss
        """
        # Normalise both embeddings and class weights onto unit sphere
        emb  = F.normalize(features, dim=1)                  # (B, D)
        W    = F.normalize(self.weight, dim=1)                # (C, D)

        # Cosine similarity: cos(theta)
        cosine = emb @ W.T                                    # (B, C)

        # cos(theta + m) = cos(theta)*cos(m) - sin(theta)*sin(m)
        sine        = (1.0 - cosine.pow(2)).clamp(min=1e-6).sqrt()
        phi         = cosine * self.cos_m - sine * self.sin_m  # (B, C)

        # For cos(theta) < cos(pi - m), using phi causes instability;
        # fall back to cos(theta) - m*sin(pi-m)
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # One-hot mask for ground-truth class
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        # Apply margin only to the ground-truth logit
        logits = one_hot * phi + (1.0 - one_hot) * cosine    # (B, C)
        logits *= self.s                                       # scale

        return F.cross_entropy(logits, labels)

    @torch.no_grad()
    def get_embeddings(self, features: torch.Tensor) -> torch.Tensor:
        """Return L2-normalised embeddings at inference (no label needed)."""
        return F.normalize(features, dim=1)


# Usage
# backbone = ResNet50(pretrained=True)  ->  (B, 2048)
# embed_fc  = nn.Linear(2048, 512)
# arc_head  = ArcFaceHead(512, num_classes=85000, s=64, m=0.5)
#
# During training:
#   feat = embed_fc(backbone(images))
#   loss = arc_head(feat, labels)
#
# During inference (retrieval):
#   emb = arc_head.get_embeddings(embed_fc(backbone(images)))
```

### CosFace Variant (for comparison)

```python
def cosface_loss(cosine: torch.Tensor, labels: torch.Tensor, s: float = 64.0, m: float = 0.35) -> torch.Tensor:
    """cosine: (B, C) dot products on unit sphere."""
    one_hot = torch.zeros_like(cosine)
    one_hot.scatter_(1, labels.view(-1, 1), 1.0)
    logits = (cosine - one_hot * m) * s
    return F.cross_entropy(logits, labels)
```

---

## 6. Supervised Contrastive Learning (SupCon)

Khosla et al. (2020). Extends SimCLR's self-supervised NT-Xent loss to the **supervised** setting where multiple images per class exist in the batch.

### Key Idea

For each anchor `i`, **all samples from the same class** in the batch are positives (not just one augmented view). This provides a richer training signal.

### Formula

```
L_i = -1/|P(i)| * sum_{p in P(i)} log [
    exp(z_i . z_p / tau)
    / sum_{a in A(i)} exp(z_i . z_a / tau)
]

P(i) = set of positives for anchor i (same class, excluding self)
A(i) = all other samples in batch
```

### Code

```python
import torch
import torch.nn.functional as F


def supervised_contrastive_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    embeddings : (B, D) L2-normalised
    labels     : (B,)  integer class ids
    """
    B = embeddings.size(0)
    device = embeddings.device

    # Similarity matrix scaled by temperature
    sim = (embeddings @ embeddings.T) / temperature         # (B, B)

    # Masks
    labels_col = labels.unsqueeze(1)                        # (B, 1)
    labels_row = labels.unsqueeze(0)                        # (1, B)
    pos_mask = (labels_col == labels_row).float()           # (B, B) — same class
    self_mask = torch.eye(B, device=device)
    pos_mask = pos_mask - self_mask                         # exclude self

    # Denominator: sum over all pairs except self
    exp_sim = torch.exp(sim) * (1 - self_mask)             # (B, B)
    log_denom = torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-9)  # (B, 1)

    # Per-anchor loss: mean over positives
    n_positives = pos_mask.sum(dim=1)                       # (B,)
    loss_per_anchor = -(pos_mask * (sim - log_denom)).sum(dim=1) / (n_positives + 1e-9)

    # Only include anchors that have at least one positive
    valid = n_positives > 0
    return loss_per_anchor[valid].mean()


# Recommended two-stage training:
# Stage 1: Train backbone + projection head with SupCon loss
# Stage 2: Freeze backbone, train a linear/ArcFace head
```

### Advantages over Triplet

- No mining needed — all positives in the batch are used.
- Scales well with batch size.
- Works with both augmentation (self-supervised) and label (supervised) positives.

---

## 7. SimCLR Revisited for Retrieval

Chen et al. (2020). Self-supervised: positives are **two augmented views** of the same image; all other images in the batch are negatives.

### NT-Xent Loss

```
L_i = -log [ exp(sim(z_i, z_j) / tau) / sum_{k≠i} exp(sim(z_i, z_k) / tau) ]

sim(u, v) = u^T v / (||u|| ||v||)   (cosine similarity)
```

For a batch of N images → 2N augmented views. Each view has exactly one positive.

### Adapting SimCLR for Retrieval

1. Remove the projection head after pre-training — use the encoder `f` directly.
2. Alternatively keep the projection head and treat it as the retrieval embedding; empirically the penultimate layer is better for retrieval.
3. Fine-tune with ArcFace or SupCon using labelled data for higher precision.

```python
def nt_xent_loss(z: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    """
    z : (2B, D) — first B rows are view-1, next B rows are view-2, L2-normalised.
    Positive pair for sample i (in 0..B-1) is sample i+B.
    """
    B2 = z.size(0)
    B  = B2 // 2
    device = z.device

    sim = (z @ z.T) / temperature                           # (2B, 2B)

    # Positive indices: i <-> i+B  (and i+B <-> i)
    labels = torch.cat([torch.arange(B, B2), torch.arange(B)]).to(device)  # (2B,)

    # Mask out self-similarity
    mask = torch.eye(B2, dtype=torch.bool, device=device)
    sim.masked_fill_(mask, float('-inf'))

    return F.cross_entropy(sim, labels)
```

---

## 8. Image Retrieval Pipeline with FAISS

A production retrieval system has three stages: **embedding extraction**, **indexing**, and **query**.

```
Query image -> Backbone -> L2-normalise -> FAISS index -> Top-K ids -> Re-rank (optional)
```

### FAISS Index Types

| Index | Speed | Memory | Accuracy | When to use |
|---|---|---|---|---|
| `IndexFlatL2` | Slow (exact) | High | Exact | Small gallery (<1M) |
| `IndexIVFFlat` | Fast | Medium | ~99% | 1M–100M vectors |
| `IndexIVFPQ` | Very fast | Low | ~95% | 100M+ vectors |
| `IndexHNSWFlat` | Very fast | Medium | ~99% | Low-latency serving |

### Full Pipeline Code

```python
import numpy as np
import faiss
import torch
import torch.nn.functional as F


# --- Step 1: Extract gallery embeddings ---

def extract_embeddings(model: torch.nn.Module, loader, device: str = "cuda") -> tuple[np.ndarray, list]:
    """
    Returns:
        embeddings : (N, D) float32 numpy array
        ids        : list of image ids / paths
    """
    model.eval()
    all_embs = []
    all_ids  = []
    with torch.no_grad():
        for images, img_ids in loader:
            images = images.to(device)
            emb = model(images)                             # (B, D)
            emb = F.normalize(emb, dim=1)                  # L2-normalise
            all_embs.append(emb.cpu().numpy())
            all_ids.extend(img_ids)
    return np.vstack(all_embs).astype(np.float32), all_ids


# --- Step 2: Build FAISS IVF index ---

def build_ivf_index(
    embeddings: np.ndarray,
    n_list: int = 256,        # number of Voronoi cells (sqrt(N) is a good default)
    use_gpu: bool = False,
) -> faiss.Index:
    """
    IVFFlat index: fast approximate nearest-neighbour search.
    """
    D = embeddings.shape[1]
    quantizer = faiss.IndexFlatL2(D)                        # coarse quantiser
    index = faiss.IndexIVFFlat(quantizer, D, n_list, faiss.METRIC_L2)

    if use_gpu:
        res   = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)

    assert not index.is_trained
    index.train(embeddings)                                 # learn Voronoi centroids
    index.add(embeddings)                                   # add vectors
    print(f"Index contains {index.ntotal} vectors")
    return index


# --- Step 3: Query ---

def retrieve(
    index: faiss.Index,
    query_emb: np.ndarray,    # (Q, D) float32, L2-normalised
    gallery_ids: list,
    top_k: int = 10,
    n_probe: int = 32,        # cells to visit at query time (speed/accuracy tradeoff)
) -> list[list]:
    """
    Returns list of length Q; each element is a list of top_k gallery ids.
    """
    index.nprobe = n_probe
    distances, indices = index.search(query_emb, top_k)    # (Q, K)
    results = []
    for row in indices:
        results.append([gallery_ids[i] for i in row if i >= 0])
    return results


# --- Putting it together ---

def save_index(index: faiss.Index, path: str):
    faiss.write_index(index, path)

def load_index(path: str) -> faiss.Index:
    return faiss.read_index(path)


# Example usage:
# gallery_embs, gallery_ids = extract_embeddings(model, gallery_loader)
# index = build_ivf_index(gallery_embs, n_list=int(np.sqrt(len(gallery_ids))))
# save_index(index, "gallery.index")
#
# query_embs, query_ids = extract_embeddings(model, query_loader)
# results = retrieve(index, query_embs, gallery_ids, top_k=10)
```

### Re-ranking with Query Expansion

```python
def alpha_qe(query_emb: np.ndarray, top_k_embs: np.ndarray, alpha: float = 3.0) -> np.ndarray:
    """
    Average Query Expansion: average query with top-K retrieved embeddings
    weighted by their similarity^alpha.
    """
    sims = (query_emb @ top_k_embs.T) ** alpha              # (K,)
    expanded = query_emb + (sims[:, None] * top_k_embs).sum(0)
    return expanded / np.linalg.norm(expanded)
```

---

## 9. Evaluation Metrics

### Recall@K (R@K)

The fraction of queries for which the correct match appears in the top-K retrieved results.

```
R@K = (1/Q) * sum_q [ 1(correct match in top-K for query q) ]
```

```python
def recall_at_k(retrieved_ids: list[list], query_labels: list, gallery_labels: list, k: int) -> float:
    """
    retrieved_ids : list of Q lists, each containing up to k gallery indices (sorted by distance)
    query_labels  : (Q,) true class for each query
    gallery_labels: (N,) true class for each gallery item
    """
    hits = 0
    for q_idx, ranked in enumerate(retrieved_ids):
        gt_label = query_labels[q_idx]
        for gal_idx in ranked[:k]:
            if gallery_labels[gal_idx] == gt_label:
                hits += 1
                break
    return hits / len(query_labels)
```

### Mean Average Precision (mAP)

Averages the area under the precision-recall curve across all queries.

```
AP_q = (1 / R_q) * sum_{k=1}^{K} P(k) * rel(k)

mAP = (1/Q) * sum_q AP_q

P(k) = precision at rank k
rel(k) = 1 if rank-k result is relevant, else 0
R_q = total relevant items in gallery for query q
```

```python
def mean_average_precision(retrieved_ids: list[list], query_labels: list, gallery_labels: list) -> float:
    aps = []
    for q_idx, ranked in enumerate(retrieved_ids):
        gt_label = query_labels[q_idx]
        n_relevant = sum(1 for g in gallery_labels if g == gt_label)
        if n_relevant == 0:
            continue
        hits = 0
        precision_sum = 0.0
        for rank, gal_idx in enumerate(ranked, start=1):
            if gallery_labels[gal_idx] == gt_label:
                hits += 1
                precision_sum += hits / rank
        aps.append(precision_sum / n_relevant)
    return float(np.mean(aps)) if aps else 0.0
```

### TAR@FAR (Face Verification)

Used for 1:1 verification. Plot the True Accept Rate vs. False Accept Rate curve and report TAR at a fixed FAR operating point (e.g., FAR=1e-4).

```python
from sklearn.metrics import roc_curve

def tar_at_far(scores: np.ndarray, labels: np.ndarray, target_far: float = 1e-4) -> float:
    """
    scores : (N,) similarity scores (higher = more similar)
    labels : (N,) 1 = genuine pair, 0 = impostor pair
    """
    fpr, tpr, _ = roc_curve(labels, scores)
    # Interpolate TAR at the desired FAR
    tar = float(np.interp(target_far, fpr, tpr))
    return tar
```

### Benchmark Datasets

| Dataset | Task | Metric |
|---|---|---|
| LFW | Face verification | Accuracy (@ threshold) |
| IJB-C | Face 1:N | TAR@FAR=1e-4 |
| CUB-200 | Fine-grained retrieval | R@1, R@2, R@4, mAP |
| Stanford Online Products | Product retrieval | R@1, R@10, R@100 |
| Market-1501 | Person Re-ID | mAP, Rank-1 |
| DukeMTMC | Person Re-ID | mAP, Rank-1 |

---

## 10. Person Re-Identification (Re-ID)

Cross-camera identity matching: given a **probe** image of a person from camera A, retrieve images of the same person from other cameras in a gallery.

### Challenges

| Challenge | Description |
|---|---|
| Viewpoint variation | 180-degree body rotation between cameras |
| Illumination | Different lighting conditions across camera feeds |
| Occlusion | Partial body visibility (bags, crowds) |
| Resolution | Low-quality footage from distant cameras |
| Clothing change | Long-term Re-ID across days |
| Background clutter | Similar backgrounds confuse appearance |

### Standard Pipeline

```
Camera A probe -> Backbone -> Embedding
Camera B gallery -> Backbone -> Embeddings -> FAISS index
                                                   |
                              Cosine / L2 distance ranking
                                                   |
                              Re-rank (e.g., k-reciprocal re-ranking)
```

### Loss Combination (state of practice)

Most competitive Re-ID models combine:
1. **ID loss** (CrossEntropy or ArcFace) — coarse class discrimination
2. **Triplet loss** — fine-grained distance constraint
3. **Center loss** (optional) — minimise intra-class variance

```python
def reid_loss(
    id_logits: torch.Tensor,    # (B, num_ids)
    embeddings: torch.Tensor,   # (B, D) L2-normalised
    labels: torch.Tensor,       # (B,)
    lambda_triplet: float = 1.0,
) -> torch.Tensor:
    id_loss  = F.cross_entropy(id_logits, labels)
    tri_loss = online_triplet_loss(embeddings, labels, margin=0.3, mining="semihard")
    return id_loss + lambda_triplet * tri_loss
```

### K-Reciprocal Re-ranking (Zhong et al. 2017)

Refines initial ranking by considering whether two images mutually retrieve each other in top-K. Commonly gives +3–5% mAP at no extra training cost.

Key idea: if image A retrieves B and B also retrieves A in its top-K, they are **k-reciprocal** neighbours — strong evidence for same identity.

---

## 11. Key Interview Points

**Q: Why not use standard cross-entropy for face recognition?**
Softmax learns a closed set of classes. At inference you need to recognise unseen identities — softmax logits are meaningless outside training classes. Metric learning produces embeddings that generalise to new identities via distance comparison.

**Q: What is the curse of dimensionality in retrieval?**
In high dimensions, L2 distances concentrate — the ratio of max to min distance approaches 1, making nearest-neighbour search less meaningful. Remedies: dimensionality reduction (PCA whitening), cosine similarity on unit sphere, use of learned metrics.

**Q: Contrastive vs. triplet loss — which is better?**
Triplet loss is generally stronger because it explicitly models relative ordering (anchor closer to positive than negative). Contrastive loss treats each pair independently. However, triplet loss requires careful mining and is sensitive to batch composition. ArcFace is superior to both for closed-set training.

**Q: Why does ArcFace outperform triplet loss?**
ArcFace: (1) uses all training samples every iteration (no mining), (2) maximises inter-class angular distance geometrically on a hypersphere, (3) has stable training with cross-entropy. Triplet depends on informative triplet availability.

**Q: What is semi-hard negative mining and why is it used?**
Hard negatives (closer than positives) cause gradient instability and collapsed embeddings early in training. Easy negatives (already beyond margin) contribute zero gradient. Semi-hard negatives — beyond the positive distance but still within the margin — provide a stable, informative gradient throughout training.

**Q: How do you choose FAISS index type in production?**
- `<1M vectors`: `IndexFlatL2` (exact)
- `1M–50M`: `IndexIVFFlat` with `n_list ~ sqrt(N)`, `n_probe ~ n_list/8`
- `>50M`: `IndexIVFPQ` for memory compression, or `IndexHNSWFlat` for low latency

**Q: What is the effect of temperature in contrastive losses?**
Low temperature (e.g., 0.07) sharpens the distribution — pushes the model to be very confident about positives. Too low causes training instability. Higher temperature (0.5) gives softer gradients but may not discriminate fine-grained classes.

**Q: How does SupCon differ from SimCLR?**
SimCLR has exactly one positive per anchor (the other augmented view). SupCon uses all same-class samples in the batch as positives, providing a denser and more informative gradient signal.

**Q: How do you handle query-gallery overlap in evaluation?**
Remove the query image itself from gallery results (using index exclusion or masking the diagonal). For Market-1501 and similar benchmarks this is handled by evaluating only cross-camera results.

**Q: What is mAP vs. Recall@K — when is each preferred?**
R@K is a binary "did you find it?" metric — good for understanding top-K coverage. mAP rewards ranking all relevant items high, not just finding one — more informative when there are multiple correct matches per query (Re-ID, product retrieval).

---
