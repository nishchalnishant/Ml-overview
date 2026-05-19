# Metric Learning

> Learn an embedding space where similar items are close and dissimilar items are far, enabling similarity search, verification, and few-shot recognition without retraining a classifier.

---

## Table of Contents

1. [What is Metric Learning](#1-what-is-metric-learning)
2. [Contrastive Loss](#2-contrastive-loss)
3. [Triplet Loss and Semi-Hard Negative Mining](#3-triplet-loss-and-semi-hard-negative-mining)
4. [N-pair / Multi-class N-pair Loss](#4-n-pair--multi-class-n-pair-loss)
5. [ArcFace / CosFace / SphereFace](#5-arcface--cosface--sphereface)
6. [Supervised Contrastive Learning (SupCon)](#6-supervised-contrastive-learning-supcon)
7. [SimCLR Revisited for Retrieval](#7-simclr-revisited-for-retrieval)
8. [Image Retrieval Pipeline with FAISS](#8-image-retrieval-pipeline-with-faiss)
9. [Evaluation Metrics](#9-evaluation-metrics)
10. [Person Re-Identification (Re-ID)](#10-person-re-identification-re-id)
11. [Key Interview Points](#11-key-interview-points)

---

## 1. What is Metric Learning

**The problem:** a standard classifier trained with softmax produces a fixed set of output classes. At inference time you need to recognize an identity that was never in the training set — a new person's face, a new product SKU, a new rare species. The softmax logits for unseen classes are undefined.

**The core insight:** instead of learning a class → score mapping, learn a mapping from inputs to a metric space where *distance reflects semantic similarity*. Two images of the same person should be close; two images of different people should be far. At inference, recognition becomes a nearest-neighbor query — no retraining needed.

```
d(f(x_i), f(x_j)) small   if x_i, x_j are same class
d(f(x_i), f(x_j)) large   if x_i, x_j are different classes
```

### Core Applications

| Domain | Task |
|---|---|
| Face verification | Is this the same person? (1:1 match) |
| Face identification | Who is this? (1:N search) |
| Image retrieval | Find visually similar images in a gallery |
| Few-shot learning | Classify new classes from 1–5 labeled examples |
| Product search | Find same/similar SKUs from a photo |
| Person Re-ID | Match pedestrians across cameras |

---

## 2. Contrastive Loss

**The problem:** you have labeled pairs of images — same class or different class. How do you train an embedding so that same-class pairs cluster together and different-class pairs are pushed apart?

**The core insight:** directly penalize same-class pairs for being far apart, and different-class pairs for being too close. Use a margin threshold: once a negative pair is already farther than the margin, it contributes zero loss — you don't penalize pairs that are already well-separated.

**The mechanics (Hadsell et al., 2006):**

```
L = y · d² + (1−y) · max(0, margin − d)²

d = ‖f(x_i) − f(x_j)‖_2
y = 1 (same class), 0 (different class)
```

```python
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, emb_a, emb_b, labels):
        """emb_a, emb_b: (B, D); labels: (B,) float, 1=same, 0=different"""
        dist = F.pairwise_distance(emb_a, emb_b, p=2)
        pos_loss = labels * dist.pow(2)
        neg_loss = (1 - labels) * F.relu(self.margin - dist).pow(2)
        return (pos_loss + neg_loss).mean()
```

**What breaks:**

- **Collapsed negatives:** most random pairs are easy negatives already beyond the margin — they contribute zero gradient and waste computation.
- **Pair construction cost:** enumerating all valid pairs is O(N²) in dataset size.
- **Margin sensitivity:** the right margin depends on embedding scale and is dataset-specific; wrong values make the loss uninformative.

---

## 3. Triplet Loss and Semi-Hard Negative Mining

**The problem:** contrastive loss treats each pair independently. It says "this pair should be close" or "this pair should be far." It doesn't express the *relative* requirement: an anchor should be closer to its positive than to any negative. Violations of this relative ordering are what actually matter for retrieval quality.

**The core insight:** train on triplets (anchor, positive, negative) and directly penalize cases where the negative is closer to the anchor than the positive. The margin enforces a buffer between them: the network is only satisfied when negatives are at least margin farther than positives.

**The mechanics (Schroff et al., FaceNet 2015):**

```
L = max(0, d(a, p) − d(a, n) + margin)
```

The loss is zero when d(a, n) ≥ d(a, p) + margin.

**Mining matters enormously:**

- **Easy negatives** (d(a,n) >> d(a,p)): already violate no constraint — zero gradient, wasted computation.
- **Hard negatives** (d(a,n) < d(a,p)): cause large gradient but collapse embeddings early in training if encountered exclusively.
- **Semi-hard negatives:** `d(a,p) < d(a,n) < d(a,p) + margin` — the constraint is violated but the negative is not harder than the positive. Stable, informative gradient throughout training.

```python
def pairwise_distances(embeddings):
    """All-pairs L2 distances. (B, B)"""
    dot = embeddings @ embeddings.T
    sq  = dot.diagonal().unsqueeze(1)
    return F.relu(sq + sq.T - 2 * dot).sqrt()

def online_triplet_loss(embeddings, labels, margin=0.3, mining="semihard"):
    """embeddings: (B, D) L2-normalized; labels: (B,) integer class ids"""
    dist = pairwise_distances(embeddings)
    B = embeddings.size(0)
    labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)
    not_self   = ~torch.eye(B, dtype=torch.bool, device=embeddings.device)
    pos_mask   = labels_eq & not_self
    neg_mask   = ~labels_eq

    d_ap = (dist * pos_mask.float()).max(dim=1).values  # hardest positive per anchor

    loss_vals = []
    for i in range(B):
        d_pos     = d_ap[i]
        neg_dists = dist[i][neg_mask[i]]
        if mining == "semihard":
            mask = (neg_dists > d_pos) & (neg_dists < d_pos + margin)
            candidates = neg_dists[mask] if mask.any() else neg_dists
        else:  # hard or all
            candidates = neg_dists
        d_neg = candidates.min()
        loss_vals.append(F.relu(d_pos - d_neg + margin))
    return torch.stack(loss_vals).mean()
```

**What breaks:**

- **Batch composition dependency:** semi-hard negatives only exist if each batch contains multiple classes and multiple examples per class. Batches too small or too class-homogeneous starve the miner.
- **Collapsed training:** `fraction_positive_triplets` approaching zero signals either convergence or mode collapse — monitor it.
- **Slow per-epoch learning:** triplets enumerate 3-tuples; with N samples there are O(N³) possible triplets. Online mining from a batch samples a tiny fraction.

---

## 4. N-pair / Multi-class N-pair Loss

**The problem:** triplet loss uses one negative at a time. When you have many negative classes in a batch, using only the hardest (or a sampled) negative discards information from all other negatives. Each triplet update makes progress against one alternative — the model could easily be fooled by a different negative.

**The core insight:** for a given anchor, use *all* other classes in the batch as negatives simultaneously. Cast it as softmax cross-entropy over inner products — the positive class should score highest against all negatives at once.

**The mechanics (Kang et al., 2016):**

Sample N classes, 2 examples each: {(a_i, p_i)}. For anchor a_i, positive is p_i; negatives are all {p_j, j≠i}.

```
L = (1/N) Σ_i log(1 + Σ_{j≠i} exp(a_i·p_j − a_i·p_i))
```

This is equivalent to softmax cross-entropy with inner-product logits.

```python
def npair_loss(anchors, positives, l2_reg=0.002):
    """anchors, positives: (N, D) L2-normalized"""
    logits  = anchors @ positives.T                # (N, N)
    targets = torch.arange(logits.size(0), device=logits.device)
    loss    = F.cross_entropy(logits, targets)
    reg     = l2_reg * (anchors.norm(dim=1).pow(2).mean()
                        + positives.norm(dim=1).pow(2).mean())
    return loss + reg
```

NT-Xent (SimCLR) is N-pair loss with temperature scaling: `logits /= τ`.

**What breaks:** sampling N classes × 2 examples per batch requires dataset balancing. If classes are highly imbalanced, the loss is biased toward frequent classes even within a batch.

---

## 5. ArcFace / CosFace / SphereFace

**The problem:** triplet and contrastive losses require mining — finding informative pairs or triplets. Mining is computationally expensive, training is sensitive to mining quality, and only a fraction of possible pairs contribute gradient. Is there a formulation that uses every training sample at every step?

**The core insight:** treat the final classification weight vectors as *class prototypes on a unit hypersphere*. Project both embeddings and prototypes onto the sphere (L2-normalize). Now the classification score is the cosine similarity (= angle) between embedding and prototype. Inject a margin into the angular domain — require the correct class to score better than all others by at least a fixed angle. This is standard softmax cross-entropy in angular space, using every sample every step, no mining required.

**Family overview:**

| Method | Margin type | Decision boundary |
|---|---|---|
| SphereFace (Liu 2017) | Multiplicative angular | cos(m·θ_yi) > cos(θ_j) |
| CosFace (Wang 2018) | Additive cosine | cos(θ_yi) − m > cos(θ_j) |
| ArcFace (Deng 2019) | Additive angular | cos(θ_yi + m) > cos(θ_j) |

ArcFace is dominant — the additive angular margin has a clean geometric interpretation on the unit sphere.

**ArcFace mechanics:**

```
L = −log [ exp(s·cos(θ_yi + m)) / (exp(s·cos(θ_yi + m)) + Σ_{j≠yi} exp(s·cos(θ_j))) ]

θ_yi = arccos(W_yi^T · f)   (both W and f are L2-normalized)
s = scale (typically 64),   m = angular margin (typically 0.5 rad ≈ 28.6°)
```

```python
class ArcFaceHead(nn.Module):
    def __init__(self, in_features, num_classes, s=64.0, m=0.5):
        super().__init__()
        self.s, self.m = s, m
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th    = math.cos(math.pi - m)
        self.mm    = math.sin(math.pi - m) * m

    def forward(self, features, labels):
        emb    = F.normalize(features, dim=1)
        W      = F.normalize(self.weight, dim=1)
        cosine = emb @ W.T                              # (B, C)
        sine   = (1.0 - cosine.pow(2)).clamp(min=1e-6).sqrt()
        phi    = cosine * self.cos_m - sine * self.sin_m
        phi    = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        logits = (one_hot * phi + (1.0 - one_hot) * cosine) * self.s
        return F.cross_entropy(logits, labels)

    @torch.no_grad()
    def get_embeddings(self, features):
        return F.normalize(features, dim=1)
```

**What breaks:**

- **Large num_classes memory:** the weight matrix W is (C × D). At C=1M identities and D=512, W is a 2 GB parameter. Requires distributed class sharding in very large-scale face recognition.
- **Margin tuning:** m=0.5 is standard for faces. For fine-grained retrieval with higher intra-class variance, a smaller margin avoids over-constraining the embedding.

---

## 6. Supervised Contrastive Learning (SupCon)

**The problem:** SimCLR's self-supervised contrastive loss has exactly one positive per anchor — the other augmented view. In a supervised setting where you have labels, many images of the same class are in each batch. Using only one positive wastes available supervision.

**The core insight:** in the supervised setting, *all* same-class samples in the batch are valid positives. Extend the NT-Xent loss to average over all of them. This produces a denser, more informative gradient signal than triplet or pair-based methods — proportional to the number of co-class samples per batch.

**The mechanics (Khosla et al., 2020):**

```
L_i = −(1/|P(i)|) Σ_{p∈P(i)} log [
    exp(z_i · z_p / τ)
    / Σ_{a∈A(i)} exp(z_i · z_a / τ)
]

P(i) = same-class samples in batch (excluding self)
A(i) = all other samples in batch
```

```python
def supervised_contrastive_loss(embeddings, labels, temperature=0.07):
    """embeddings: (B, D) L2-normalized; labels: (B,) integer class ids"""
    B      = embeddings.size(0)
    sim    = (embeddings @ embeddings.T) / temperature      # (B, B)
    pos_mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
    self_mask = torch.eye(B, device=embeddings.device)
    pos_mask  = pos_mask - self_mask

    exp_sim   = torch.exp(sim) * (1 - self_mask)
    log_denom = torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-9)

    n_pos     = pos_mask.sum(dim=1)
    loss_per  = -(pos_mask * (sim - log_denom)).sum(dim=1) / (n_pos + 1e-9)
    valid     = n_pos > 0
    return loss_per[valid].mean()
```

**What breaks:** requires large batches with many classes per batch to ensure each anchor has multiple positives. Memory scales quadratically with batch size for the similarity matrix. Performance is sensitive to temperature: too small collapses gradients; too large is uninformative.

---

## 7. SimCLR Revisited for Retrieval

**The problem:** labeled data for metric learning is expensive — labeling all identities for face recognition or all SKUs for product search requires enormous annotation effort. Can you learn good metric embeddings without labels?

**The core insight:** self-supervision via data augmentation. Two differently augmented crops of the same image are semantically the same thing — treat them as a positive pair. All other images in the batch are negatives. The model must learn representations that are invariant to augmentation but discriminative across images.

**The mechanics (Chen et al., SimCLR 2020):**

```
L_i = −log [ exp(sim(z_i, z_j) / τ) / Σ_{k≠i} exp(sim(z_i, z_k) / τ) ]
sim(u, v) = uᵀv / (‖u‖‖v‖)
```

For a batch of N images → 2N augmented views. Each view has exactly one positive (the other augmented view of the same image).

```python
def nt_xent_loss(z, temperature=0.5):
    """z: (2B, D) — first B rows are view-1, next B rows are view-2, L2-normalized"""
    B2 = z.size(0); B = B2 // 2
    sim = (z @ z.T) / temperature                           # (2B, 2B)
    labels = torch.cat([torch.arange(B, B2), torch.arange(B)]).to(z.device)
    sim.masked_fill_(torch.eye(B2, dtype=torch.bool, device=z.device), float('-inf'))
    return F.cross_entropy(sim, labels)
```

**Adapting for retrieval:** remove the projection head after pre-training — the backbone encoder f is used directly. Fine-tune with ArcFace or SupCon using labeled data for higher retrieval precision.

**What breaks:** without labels, SimCLR cannot distinguish same-class negatives (two different images of the same person) from true negatives. This limits retrieval precision on fine-grained tasks. SupCon fine-tuning addresses this.

---

## 8. Image Retrieval Pipeline with FAISS

**The problem:** after training a metric embedding, how do you efficiently find the nearest neighbors of a query among millions of gallery embeddings? Exact nearest-neighbor search is O(N·D) per query — at N=100M, D=512 this is 51 GFLOP per query.

**The core insight:** approximate nearest-neighbor search via an inverted file index (IVF). Cluster the gallery into Voronoi cells; at query time, only search the nearest cells. Accuracy is traded for speed via the number of cells probed.

```
Query → L2-normalize → FAISS index → top-K ids → optional re-ranking
```

### FAISS index types

| Index | Speed | Memory | Accuracy | When to use |
|---|---|---|---|---|
| `IndexFlatL2` | Slow (exact) | High | Exact | <1M vectors |
| `IndexIVFFlat` | Fast | Medium | ~99% | 1M–100M vectors |
| `IndexIVFPQ` | Very fast | Low | ~95% | >100M vectors |
| `IndexHNSWFlat` | Very fast | Medium | ~99% | Low-latency serving |

```python
def extract_embeddings(model, loader, device="cuda"):
    model.eval()
    all_embs, all_ids = [], []
    with torch.no_grad():
        for images, img_ids in loader:
            emb = F.normalize(model(images.to(device)), dim=1)
            all_embs.append(emb.cpu().numpy())
            all_ids.extend(img_ids)
    return np.vstack(all_embs).astype(np.float32), all_ids

def build_ivf_index(embeddings, n_list=256, use_gpu=False):
    D = embeddings.shape[1]
    quantizer = faiss.IndexFlatL2(D)
    index = faiss.IndexIVFFlat(quantizer, D, n_list, faiss.METRIC_L2)
    if use_gpu:
        index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, index)
    index.train(embeddings)
    index.add(embeddings)
    return index

def retrieve(index, query_emb, gallery_ids, top_k=10, n_probe=32):
    index.nprobe = n_probe
    distances, indices = index.search(query_emb, top_k)
    return [[gallery_ids[i] for i in row if i >= 0] for row in indices]
```

**Re-ranking with Alpha Query Expansion:**
```python
def alpha_qe(query_emb, top_k_embs, alpha=3.0):
    """Average query with top-K retrieved embeddings weighted by similarity^alpha"""
    sims = (query_emb @ top_k_embs.T) ** alpha
    expanded = query_emb + (sims[:, None] * top_k_embs).sum(0)
    return expanded / np.linalg.norm(expanded)
```

**What breaks:** IVF accuracy degrades when the number of probed cells is too small. Setting n_probe too low also causes the effective recall@1 to fall below useful thresholds; the tradeoff curve must be profiled on the target dataset.

---

## 9. Evaluation Metrics

### Recall@K (R@K)

Fraction of queries for which the correct match appears in the top-K retrieved results:
```
R@K = (1/Q) Σ_q 1(correct match in top-K for query q)
```

### Mean Average Precision (mAP)

Averages the area under the precision-recall curve across all queries:
```
AP_q = (1/R_q) Σ_{k=1}^K P(k) · rel(k)
mAP  = (1/Q)   Σ_q AP_q
```

R_q = total relevant items in gallery for query q.

```python
def mean_average_precision(retrieved_ids, query_labels, gallery_labels):
    aps = []
    for q_idx, ranked in enumerate(retrieved_ids):
        gt = query_labels[q_idx]
        n_rel = sum(1 for g in gallery_labels if g == gt)
        if not n_rel: continue
        hits = 0; psum = 0.0
        for rank, gal_idx in enumerate(ranked, 1):
            if gallery_labels[gal_idx] == gt:
                hits += 1; psum += hits / rank
        aps.append(psum / n_rel)
    return float(np.mean(aps)) if aps else 0.0
```

### TAR@FAR (Face Verification)

For 1:1 verification: report the True Accept Rate at a fixed False Accept Rate operating point (e.g., TAR@FAR=1e-4).

```python
from sklearn.metrics import roc_curve

def tar_at_far(scores, labels, target_far=1e-4):
    """scores: (N,) similarity; labels: (N,) 1=genuine, 0=impostor"""
    fpr, tpr, _ = roc_curve(labels, scores)
    return float(np.interp(target_far, fpr, tpr))
```

### Benchmark Datasets

| Dataset | Task | Metric |
|---|---|---|
| LFW | Face verification | Accuracy (at threshold) |
| IJB-C | Face 1:N | TAR@FAR=1e-4 |
| CUB-200 | Fine-grained retrieval | R@1, R@2, R@4, mAP |
| Stanford Online Products | Product retrieval | R@1, R@10, R@100 |
| Market-1501 | Person Re-ID | mAP, Rank-1 |

---

## 10. Person Re-Identification (Re-ID)

**The problem:** given a probe image of a person from camera A, find images of the same person from other cameras. The challenge: the same person looks drastically different from different angles, under different lighting, at different distances, and across different days.

### Challenges

| Challenge | Description |
|---|---|
| Viewpoint variation | Up to 180° rotation between cameras |
| Illumination | Different lighting across feeds |
| Occlusion | Partial body visibility (bags, crowds) |
| Resolution | Low-quality footage from distant cameras |
| Clothing change | Long-term Re-ID across days |
| Background clutter | Similar backgrounds confuse appearance |

### Standard pipeline

```
Camera A probe → backbone → embedding
Camera B gallery → backbone → embeddings → FAISS index → ranking
                                                → k-reciprocal re-ranking
```

### Loss combination

Most competitive Re-ID models combine:
1. **ID loss** (CrossEntropy or ArcFace): coarse class discrimination.
2. **Triplet loss**: fine-grained distance constraint.
3. **Center loss** (optional): minimizes intra-class variance.

```python
def reid_loss(id_logits, embeddings, labels, lambda_triplet=1.0):
    id_loss  = F.cross_entropy(id_logits, labels)
    tri_loss = online_triplet_loss(embeddings, labels, margin=0.3, mining="semihard")
    return id_loss + lambda_triplet * tri_loss
```

### K-Reciprocal Re-ranking (Zhong et al. 2017)

**The problem:** the initial ranking from cosine distance is imperfect. Two images of the same person should *mutually* retrieve each other near the top. If A retrieves B in its top-K, and B also retrieves A in its top-K, that's strong evidence for a match.

Re-ranking uses k-reciprocal neighborhoods: if A and B are k-reciprocal neighbors (each appears in the other's top-K), their distance is reduced. Typically gives +3–5% mAP at no extra training cost.

---

## 11. Key Interview Points

**Q: Why not use standard cross-entropy for face recognition?**
Softmax learns a closed set of classes. At inference you need to recognize unseen identities — softmax logits have no meaning outside training classes. Metric learning produces embeddings that generalize to new identities via distance comparison.

**Q: What is the curse of dimensionality in retrieval?**
In high dimensions, pairwise L2 distances concentrate — the ratio of max to min distance approaches 1, making nearest-neighbor search less meaningful. Remedies: PCA whitening, cosine similarity on the unit sphere, learned metrics.

**Q: Contrastive vs. triplet loss — which is better?**
Triplet loss is generally stronger because it explicitly enforces relative ordering (anchor closer to positive than to negative). Contrastive loss treats each pair independently. However, triplet requires careful mining and is sensitive to batch composition. ArcFace outperforms both for closed-set training.

**Q: Why does ArcFace outperform triplet loss?**
ArcFace uses every training sample at every step (no mining), maximizes inter-class angular distance geometrically on a hypersphere, and has stable training via cross-entropy. Triplet depends on finding informative triplets — a small, variable fraction of the batch.

**Q: What is semi-hard negative mining and why is it used?**
Hard negatives (closer than positives) cause gradient instability and collapsed embeddings early in training. Easy negatives (already beyond the margin) contribute zero gradient. Semi-hard negatives — beyond the positive distance but still within the margin — provide a stable, informative gradient throughout training.

**Q: How do you choose the FAISS index type in production?**
- <1M vectors: `IndexFlatL2` (exact search).
- 1M–50M: `IndexIVFFlat` with n_list ~ √N, n_probe ~ n_list/8.
- >50M: `IndexIVFPQ` for memory compression, or `IndexHNSWFlat` for low latency.

**Q: What is the effect of temperature in contrastive losses?**
Low temperature (e.g., 0.07) sharpens the distribution — pushes the model to be very confident about positives vs. negatives. Too low causes training instability (extreme gradients). Higher temperature (0.5) gives softer gradients but may not discriminate fine-grained classes.

**Q: How does SupCon differ from SimCLR?**
SimCLR has exactly one positive per anchor (the other augmented view). SupCon uses all same-class samples in the batch as positives, providing a denser and more informative gradient signal proportional to the number of co-class samples per batch.

**Q: How do you handle query-gallery overlap in evaluation?**
Remove the query image itself from gallery results. For Market-1501 and similar benchmarks, evaluate only cross-camera results — same-camera matches are excluded to avoid trivially easy positives.

**Q: What is mAP vs. Recall@K — when is each preferred?**
R@K is binary — did you find at least one correct match in the top-K? Good for understanding top-K coverage. mAP rewards ranking *all* relevant items highly, not just finding one — more informative when there are multiple correct matches per query (Re-ID, product retrieval with many SKU variants).
