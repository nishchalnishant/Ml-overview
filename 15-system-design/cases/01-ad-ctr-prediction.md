---
module: System Design
topic: System Design
subtopic: Ad Ctr Prediction
status: unread
tags: [productionml, ml, system-design-ad-ctr-predictio]
---
# Ad Click-Through Rate (CTR) Prediction System Design

End-to-end ML system for predicting the probability a user clicks an ad. Canonical system design question at Meta, Google, Twitter/X, and Criteo. Underpins the entire programmatic advertising industry — a 1% improvement in CTR prediction translates directly to hundreds of millions in annual revenue.

**Scale:** 10M+ impressions/second, <10ms scoring latency, 100B+ training examples/day, embedding tables with billions of IDs.

---

## 1. Problem Framing

### Clarifying Questions

- **Ad format?** Display (banner), search (keyword-triggered), social (feed), video pre-roll — each has different click semantics and feature sets
- **Latency budget?** Entire ad serving stack must complete in <100ms; CTR scoring gets ~10ms
- **Training data freshness?** Is 24h-stale model acceptable, or is online learning required?
- **Feedback loop?** Are post-click conversions in scope, or just clicks?
- **Sparse ID space?** How many user IDs, ad IDs, publisher IDs — determines embedding table memory
- **Multi-task?** Predict only CTR, or also CVR (conversion rate), engagement, etc.?
- **Auction type?** First-price vs second-price — calibration requirements differ

### Business Metrics vs ML Metrics

| Layer | Metric | Formula | Who cares |
|---|---|---|---|
| Business | Revenue per mille (RPM) | Revenue / 1000 impressions | Ads org, CFO |
| Business | eCPM | Bid × pCTR × quality | Ad auction team |
| Business | Advertiser ROI | Conversions / spend | Advertisers |
| ML | Log loss (NLL) | $-\frac{1}{N}\sum y\log\hat{p} + (1-y)\log(1-\hat{p})$ | Model team |
| ML | AUC-ROC | Ranking quality | Model team |
| ML | Calibration error | $\|\mathbb{E}[\hat{p}] - \mathbb{E}[y]\|$ | Auction team |

**Why calibration matters more than ranking for ads:**

AUC measures whether the model ranks a clicked ad higher than a non-clicked ad — it is a pure ranking metric. The ad auction, however, uses the *absolute value* of pCTR to compute eCPM:

$$\text{eCPM} = \text{bid} \times \hat{p}_{CTR} \times \text{quality\_score}$$

If pCTR is systematically 2× too high, every advertiser overpays by 2× and the auction allocates budget incorrectly. A model with slightly worse AUC but better calibration is often more valuable in production.

**Threshold:** Log loss improvement of 0.001 (~0.1%) is considered production-worthy at Google scale; AUC improvement of 0.001 is the corresponding ranking signal.

---

## 2. Scale Requirements

| Dimension | Number | Implication |
|---|---|---|
| Impressions/second | 10M+ | Model inference must be embarrassingly parallelizable |
| P99 scoring latency | <10ms | No sequential DB lookups at score time |
| Training examples/day | 100B+ | Distributed training, not single-machine |
| Unique user IDs | ~3B (Meta/Google scale) | Embedding table: 3B × 16 dims × 4 bytes = 192 GB |
| Unique ad IDs | ~500M active | Embedding table: 500M × 16 dims × 4 bytes = 32 GB |
| Unique (user, ad) cross features | Trillions possible | Feature hashing required, exact storage impossible |
| Model update frequency | Every 15 min–1 hour | Continuous training pipeline |
| Training throughput needed | ~1.2M examples/sec | Multi-host all-reduce or parameter server |

**Memory math for embedding tables (DLRM-style):**

```
Total sparse embedding memory =
  Σ(vocabulary_size_i × embedding_dim_i × 4 bytes)

Example (Meta DLRM paper):
  26 sparse feature tables
  Avg vocab: 10M IDs, dim=64
  = 26 × 10M × 64 × 4 bytes = 66 GB

Real Meta DLRM:
  Some tables: 10B+ IDs → does not fit on a single GPU
  → embedding tables sharded across CPU memory / multiple hosts
```

---

## 3. System Architecture

### Online Serving Path

```
User Request (page load / app open)
        │
        ▼
 ┌─────────────────┐
 │  Ad Request     │  user_id, page context, device, geo
 │  Gateway        │  <1ms
 └────────┬────────┘
          │
          ▼
 ┌─────────────────┐      ┌────────────────────┐
 │  Ad Retrieval   │◄─────│  Candidate Index   │
 │  (Recall ~1K)   │      │  (FAISS / ScaNN)   │
 │  Two-Tower NN   │      │  ~1B ads indexed   │
 └────────┬────────┘      └────────────────────┘
          │  ~1000 candidates
          ▼
 ┌─────────────────┐      ┌────────────────────┐
 │  Feature        │◄─────│  Feature Store     │
 │  Assembly       │      │  Redis (user feats)│
 │                 │◄─────│  Memcached (ad)    │
 └────────┬────────┘      └────────────────────┘
          │
          ▼
 ┌─────────────────┐
 │  CTR Scorer     │  DLRM / DCN v2 / Wide & Deep
 │  (Ranking)      │  ~1000 ads → scored in batch
 │                 │  <8ms GPU inference
 └────────┬────────┘
          │  top-K (~50) with pCTR scores
          ▼
 ┌─────────────────┐
 │  Auction Engine │  eCPM = bid × pCTR × quality
 │  (Pricing)      │  second-price auction
 │                 │  position bias applied
 └────────┬────────┘
          │
          ▼
 ┌─────────────────┐
 │  Ad Serving     │  render selected ads
 │  & Logging      │  emit impression log → Kafka
 └────────┬────────┘
          │
          ▼
      User sees ad
      Click / No-click → feedback event → Kafka
```

### Offline Training Pipeline

```
 ┌────────────────────────────────────────────────────────────┐
 │                    TRAINING PIPELINE                       │
 │                                                            │
 │  Kafka (click/impression events)                           │
 │        │                                                   │
 │        ▼                                                   │
 │  ┌──────────────┐    feature join    ┌──────────────────┐  │
 │  │  Log         │───────────────────►│  Training        │  │
 │  │  Collection  │                    │  Examples        │  │
 │  │  (Flink)     │◄───────────────────│  (Parquet/ORC)   │  │
 │  └──────────────┘   user/ad features └────────┬─────────┘  │
 │                      from offline store        │            │
 │                                                ▼            │
 │  ┌──────────────┐                   ┌──────────────────┐   │
 │  │  Evaluation  │◄──────────────────│  Distributed     │   │
 │  │  & Holdout   │   trained model   │  Training        │   │
 │  │  Validation  │                   │  (PyTorch DDP /  │   │
 │  └──────┬───────┘                   │  Parameter       │   │
 │         │                           │  Server)         │   │
 │         │ passes eval               └──────────────────┘   │
 │         ▼                                                   │
 │  ┌──────────────┐                                          │
 │  │  Shadow       │  canary on 1% traffic → compare metrics │
 │  │  Deployment  │                                          │
 │  └──────┬───────┘                                          │
 │         ▼                                                   │
 │  ┌──────────────┐                                          │
 │  │  Production  │  gradual rollout: 1% → 10% → 100%        │
 │  │  Serving     │                                          │
 │  └──────────────┘                                          │
 └────────────────────────────────────────────────────────────┘
```

---

## 4. Feature Engineering

### User Features

| Feature | Type | Computation | Signal |
|---|---|---|---|
| user_id | Sparse ID | Embedding lookup | Personalization |
| age_bucket, gender | Dense | Direct | Demographic targeting |
| interest_embedding | Dense 64d | Averaged item embeddings from past 30d activity | Interest vector |
| historical_ctr_7d | Dense | clicks / impressions last 7 days | User engagement level |
| historical_ctr_by_category_7d | Dense | Per-category click rates | Category affinity |
| device_type | Sparse ID | Embedding | Mobile vs desktop behavior |
| hour_of_day, day_of_week | Dense (cyclic) | sin/cos encoding | Temporal patterns |
| recency_last_click | Dense | seconds since last ad click | Engagement freshness |

### Ad Features

| Feature | Type | Computation | Signal |
|---|---|---|---|
| ad_id | Sparse ID | Embedding lookup | Ad-specific quality |
| advertiser_id | Sparse ID | Embedding lookup | Advertiser brand effect |
| creative_embedding | Dense 128d | CNN/CLIP on image/text | Visual/text quality |
| ad_historical_ctr | Dense | clicks / impressions (lifetime) | Baseline ad quality |
| ad_ctr_by_placement | Dense | Per-placement click rates | Placement affinity |
| topic_category | Sparse ID | Taxonomy embedding | Content relevance |
| bid_price_normalized | Dense | log(bid) | Price signal |

### Context Features

| Feature | Type | Signal |
|---|---|---|
| placement_id | Sparse ID | Above-fold vs below-fold, sidebar vs feed |
| page_topic_embedding | Dense | Page content relevance |
| query_embedding (search) | Dense 128d | Intent signal (most powerful for search ads) |
| geo_dma | Sparse ID | Local relevance |
| connection_type | Dense | WiFi vs cellular → video ad suitability |
| app_id / site_id | Sparse ID | Publisher-level engagement patterns |

### Cross Features

Raw features miss interactions. A young user + gaming ad + mobile context is very different from 3 independent signals.

| Cross Feature | Construction | Why |
|---|---|---|
| user_category × ad_category | Embedding product or FM | User-ad relevance |
| user_device × ad_format | Lookup table | Mobile user + video format interaction |
| user_historical_ctr × ad_ctr | Product | Both high → strong prior |
| query × ad_title (search) | Dot product of embeddings | Query-ad semantic match |

**Feature hashing for cross features:**

Cross features from high-cardinality IDs cannot be stored exactly. Feature hashing projects them into a fixed-size hash space:

```python
def hash_cross_feature(feature_a: int, feature_b: int, hash_size: int = 2**24) -> int:
    """
    Hash cross feature (feature_a, feature_b) into [0, hash_size).
    Uses MurmurHash3 for low collision rate.
    """
    import mmh3
    combined = f"{feature_a}_{feature_b}"
    return mmh3.hash(combined, signed=False) % hash_size

# At Meta/Google: hash_size = 2^26 to 2^30 depending on memory budget
# Collision rate at 2^26 with 10M unique pairs ≈ 0.015% — acceptable
```

### Embedding Table Construction

Sparse IDs (user_id, ad_id, etc.) are mapped to dense embeddings:

```python
import torch
import torch.nn as nn

class SparseFeatureEncoder(nn.Module):
    def __init__(self, vocab_sizes: dict, embedding_dim: int = 64):
        super().__init__()
        # One embedding table per sparse feature field
        self.embeddings = nn.ModuleDict({
            name: nn.EmbeddingBag(
                num_embeddings=vocab_size,
                embedding_dim=embedding_dim,
                mode='mean',         # mean-pool multi-value fields
                sparse=True          # sparse gradients → much faster for large tables
            )
            for name, vocab_size in vocab_sizes.items()
        })
    
    def forward(self, sparse_inputs: dict) -> torch.Tensor:
        embedded = [self.embeddings[name](ids) for name, ids in sparse_inputs.items()]
        return torch.cat(embedded, dim=1)
```

---

## 5. Model Architecture Evolution

### Stage 1: Logistic Regression (2005–2012)

The baseline. Fast, interpretable, deployable. Still used at some companies for cold-start or low-latency tiers.

```python
# Logistic regression on hashed features
# Trained with FTRL-Proximal (online learning friendly)
# Input: ~10^9 dimensional sparse vector (one-hot + cross features)
# L1 regularization → automatic feature selection
```

**Limitation:** Cannot learn feature interactions beyond manually engineered crosses.

### Stage 2: Gradient Boosted Trees (2012–2016)

GBDT (XGBoost, LightGBM) learns feature interactions automatically. Facebook's 2014 paper used GBDT to transform features before logistic regression:

```
Raw features → GBDT (learn interactions) → leaf indices as features → Logistic Regression
```

**Limitation:** GBDT cannot handle 10^9 sparse IDs (embedding tables). Fixed vocabulary.

### Stage 3: Wide & Deep (Google, 2016)

The canonical deep learning approach for recommendation systems.

```
                 ┌──────────────────────────────────────┐
                 │           Wide & Deep Model           │
                 │                                       │
  Dense feats ──►│  Deep Part         Wide Part          │
  Sparse IDs ──►│  (embeddings +     (memorization:     │
                 │   MLP layers)      cross features     │
                 │                    + LR)              │
                 │         └────────┬────────┘           │
                 │                  ▼                    │
                 │           sigmoid(wide + deep)        │
                 └──────────────────────────────────────┘
```

- **Wide part:** Cross-product features + logistic regression. Memorizes frequent co-occurrences (e.g., "user installed Netflix" AND "ad is for Disney+" → high CTR).
- **Deep part:** Embedding lookup + 3-layer MLP. Generalizes to unseen combinations.

### Stage 4: DeepFM (Huawei, 2017)

Replaces the wide (manual) part with a Factorization Machine that learns all pairwise interactions automatically:

$$\hat{y} = \sigma\left(\underbrace{y_{FM}}_{\text{1st + 2nd order}} + \underbrace{y_{DNN}}_{\text{high-order}}\right)$$

$$y_{FM} = w_0 + \sum_i w_i x_i + \sum_{i<j} \langle \mathbf{v}_i, \mathbf{v}_j \rangle x_i x_j$$

### Stage 5: DCN v2 — Deep & Cross Network (Google, 2021)

Explicit feature crossing at every layer, more expressive than FM's pairwise-only crossing:

```python
class CrossLayer(nn.Module):
    """DCN v2 cross layer: x_{l+1} = x_0 * (W_l x_l + b_l) + x_l"""
    def __init__(self, d: int):
        super().__init__()
        self.W = nn.Linear(d, d, bias=True)
    
    def forward(self, x0: torch.Tensor, xl: torch.Tensor) -> torch.Tensor:
        return x0 * self.W(xl) + xl  # element-wise product with residual

class DCNv2(nn.Module):
    def __init__(self, input_dim: int, n_cross_layers: int = 3, mlp_dims: list = [512, 256, 128]):
        super().__init__()
        self.cross_layers = nn.ModuleList([CrossLayer(input_dim) for _ in range(n_cross_layers)])
        mlp_layers = []
        in_d = input_dim
        for out_d in mlp_dims:
            mlp_layers += [nn.Linear(in_d, out_d), nn.ReLU()]
            in_d = out_d
        self.mlp = nn.Sequential(*mlp_layers)
        self.output = nn.Linear(input_dim + mlp_dims[-1], 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = x
        xl = x
        for cross in self.cross_layers:
            xl = cross(x0, xl)
        deep = self.mlp(x)
        combined = torch.cat([xl, deep], dim=1)
        return torch.sigmoid(self.output(combined))
```

### Stage 6: DLRM — Deep Learning Recommendation Model (Meta, 2019)

Meta's production architecture. Key insight: treat sparse (ID) and dense features separately before combining.

```
  Sparse IDs ──► Embedding Tables ──► Pooled embeddings (one per field)
                                              │
                                              ▼
                                    ┌─────────────────┐
  Dense feats ──► Bottom MLP ──────►│  Interaction    │  dot products of
                                    │  Layer          │  all embedding pairs
                                    └────────┬────────┘
                                             │
                                             ▼
                                      Top MLP → sigmoid
```

**Interaction layer** computes all pairwise dot products between embeddings:

```python
def dlrm_interaction(dense_output: torch.Tensor, sparse_embeddings: list) -> torch.Tensor:
    """
    dense_output: [B, D] from bottom MLP
    sparse_embeddings: list of [B, D] tensors
    Returns concatenated interaction features
    """
    T = torch.stack([dense_output] + sparse_embeddings, dim=1)  # [B, n+1, D]
    # All pairwise dot products
    Z = torch.bmm(T, T.transpose(1, 2))  # [B, n+1, n+1]
    # Upper triangle (no diagonal, no redundant pairs)
    n = Z.shape[1]
    indices = torch.triu_indices(n, n, offset=1)
    interactions = Z[:, indices[0], indices[1]]  # [B, n*(n-1)/2]
    return torch.cat([dense_output, interactions], dim=1)
```

### Two-Tower vs Single Model

| Use case | Architecture | Why |
|---|---|---|
| **Retrieval** (1B → 1K candidates) | Two-tower | User tower and ad tower computed independently; ad embeddings pre-indexed in FAISS. O(1) lookup at query time |
| **Ranking** (1K → 50 ads) | Single model (DLRM/DCN) | Full feature interaction between user and ad. More accurate but requires both towers to be present simultaneously |

```
Two-Tower for Retrieval:
  User features → User Tower → user_embedding (128d)
  Ad features   → Ad Tower   → ad_embedding (128d)     ← pre-computed, indexed
  
  Score = dot(user_embedding, ad_embedding)
  → ANN search returns top-1000 candidates in <5ms

Single Model for Ranking:
  [user features | ad features | context features] → joint model
  → pCTR for each candidate
  → more accurate because user-ad interactions are modeled jointly
```

---

## 6. Training at Scale

### Online Learning with FTRL (Logistic Regression tier)

Follow-The-Regularized-Leader-Proximal is the standard online optimizer for sparse CTR models. Used in production at Google for years before deep learning took over the ranking tier.

```python
class FTRLOptimizer:
    """
    FTRL-Proximal for sparse logistic regression.
    McMahan et al., 2013 — Google's production ad click model.
    """
    def __init__(self, alpha=0.1, beta=1.0, l1=1.0, l2=1.0):
        self.alpha = alpha  # learning rate scale
        self.beta = beta    # smoothing
        self.l1 = l1        # L1 regularization (induces sparsity)
        self.l2 = l2        # L2 regularization
        self.z = {}         # accumulated gradient stats
        self.n = {}         # accumulated squared gradients
    
    def update(self, feature_id: int, gradient: float) -> None:
        if feature_id not in self.n:
            self.z[feature_id] = 0.0
            self.n[feature_id] = 0.0
        
        sigma = (np.sqrt(self.n[feature_id] + gradient**2) - np.sqrt(self.n[feature_id])) / self.alpha
        self.z[feature_id] += gradient - sigma * self._get_weight(feature_id)
        self.n[feature_id] += gradient ** 2
    
    def _get_weight(self, feature_id: int) -> float:
        z_i = self.z.get(feature_id, 0.0)
        n_i = self.n.get(feature_id, 0.0)
        if abs(z_i) <= self.l1:
            return 0.0  # L1 prox → zero weight (sparse model)
        sign = 1.0 if z_i > 0 else -1.0
        return -(z_i - sign * self.l1) / (self.l2 + (self.beta + np.sqrt(n_i)) / self.alpha)
```

**FTRL properties:**
- Handles 10^9 sparse features naturally (only store touched features)
- Per-coordinate adaptive learning rates (like AdaGrad)
- L1 regularization produces truly sparse models (important for memory)
- Can process ~100K examples/second per worker

### Distributed Training for Deep Models

**Parameter Server architecture (legacy):**
```
Workers compute gradients locally
    → push gradients to Parameter Server shards
    → PS aggregates, updates weights
    → Workers pull updated weights

Problem: PS is a bandwidth bottleneck at scale
```

**All-Reduce (PyTorch DDP, preferred for dense parameters):**
```
Each worker has full model copy
    → forward + backward pass on local mini-batch
    → ring all-reduce: each worker exchanges gradients
    → all workers apply same gradient update (exact sync)

Bandwidth: O(2(N-1)/N × model_size) per step → near-linear scaling
```

**Hybrid for DLRM (Meta's approach):**
```
Dense parameters (MLP weights) → all-reduce across GPUs (DDP)
Sparse embedding tables        → model parallel, sharded across hosts
                                  (too large for single GPU)
→ Each GPU handles different embedding table shards
→ All-to-all communication for embedding lookup across shards
```

### Handling Delayed Feedback

**The problem:** A user clicks an ad, then converts (purchases) 3 days later. The training example was logged as "no conversion" if we trained immediately after the click.

**Approaches:**

| Strategy | Mechanism | Trade-off |
|---|---|---|
| Fixed attribution window | Wait 7 days before finalizing label | Simple, but 7-day data lag |
| Delayed feedback model (Chapelle, 2014) | Model conversion delay as a distribution; weight examples by likelihood of observing label | Complex, ~1-day lag |
| Elapsed-time encoding | Add "time since impression" as a feature; model learns that recent negatives are noisier | Partial solution |
| Importance weighting | Re-weight delayed positives when they arrive to correct for initial negative label | Needs careful implementation |

### Negative Sampling

Raw data is ~99% negatives (impressions without clicks). Training on all negatives wastes compute and produces poorly calibrated models.

```python
def negative_sampling_with_correction(
    positives: list,
    negatives: list,
    sampling_rate: float = 0.01  # keep 1% of negatives
) -> tuple:
    """
    Downsample negatives. Correct model output for sampling bias.
    
    Calibration correction: if q = sampling_rate, then
    p_true = p_model / (p_model + (1 - p_model) / q)
    """
    sampled_negatives = random.sample(negatives, int(len(negatives) * sampling_rate))
    
    # Correction at inference time (do NOT skip this):
    def correct_probability(p_model: float, q: float = sampling_rate) -> float:
        return p_model / (p_model + (1 - p_model) / q)
    
    return positives + sampled_negatives, correct_probability
```

**Critical:** Forgetting to apply the calibration correction after negative sampling is a common production bug. The raw model output will be ~100× too high if correction is skipped.

---

## 7. Calibration

### Why CTR Models Must Be Calibrated

Miscalibrated pCTR directly distorts auction prices and advertiser ROI.

**Example:** Advertiser bids $10 CPM. True CTR = 0.01. Model predicts pCTR = 0.02 (2× overestimate).
- Auction computes eCPM = $10 × 0.02 = $0.20
- Advertiser wins auctions they shouldn't, pays too much per click
- Budget depletes 2× faster than expected

**Measuring calibration:**

```python
def calibration_curve(y_true, y_prob, n_bins=20):
    """
    Expected Calibration Error (ECE) and calibration plot.
    Well-calibrated: predicted 0.1 → 10% actual click rate.
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_lower = bin_edges[:-1]
    bin_upper = bin_edges[1:]
    
    ece = 0.0
    calibration_data = []
    
    for lower, upper in zip(bin_lower, bin_upper):
        mask = (y_prob >= lower) & (y_prob < upper)
        if mask.sum() == 0:
            continue
        bin_accuracy = y_true[mask].mean()
        bin_confidence = y_prob[mask].mean()
        bin_weight = mask.sum() / len(y_true)
        
        ece += bin_weight * abs(bin_accuracy - bin_confidence)
        calibration_data.append((bin_confidence, bin_accuracy, mask.sum()))
    
    return ece, calibration_data

# Target: ECE < 0.002 in production CTR models
```

### Post-Hoc Calibration Methods

**Platt Scaling** (sigmoid fit on validation set):

```python
from sklearn.linear_model import LogisticRegression

def platt_scaling(val_scores, val_labels):
    """Fit a sigmoid to map raw scores → calibrated probabilities."""
    lr = LogisticRegression()
    lr.fit(val_scores.reshape(-1, 1), val_labels)
    # Production: apply lr.predict_proba(raw_score)[:, 1]
    return lr
```

**Isotonic Regression** (non-parametric, more flexible):

```python
from sklearn.isotonic import IsotonicRegression

def isotonic_calibration(val_scores, val_labels):
    """Fit a non-decreasing step function. More flexible than Platt."""
    ir = IsotonicRegression(out_of_bounds='clip')
    ir.fit(val_scores, val_labels)
    return ir

# Isotonic regression can overfit on small val sets → prefer Platt if <10K samples
```

**Temperature Scaling** (single-parameter, fast):

$$p_{calibrated} = \sigma\left(\frac{\text{logit}(p_{raw})}{T}\right)$$

```python
def find_temperature(val_logits, val_labels):
    """Find T that minimizes NLL on validation set."""
    from scipy.optimize import minimize_scalar
    
    def nll(T):
        scaled = torch.tensor(val_logits / T)
        return F.binary_cross_entropy_with_logits(
            scaled, torch.tensor(val_labels, dtype=torch.float32)
        ).item()
    
    result = minimize_scalar(nll, bounds=(0.1, 10.0), method='bounded')
    return result.x  # optimal temperature T

# T > 1 → model is overconfident (common)
# T < 1 → model is underconfident
```

**Which to use:**
- Temperature scaling: simplest, one parameter, good default
- Platt scaling: when score distribution differs from logistic
- Isotonic regression: when you have >100K calibration examples and need flexible shape

---

## 8. Auction Mechanics

### Second-Price Auction (Vickrey Auction)

In a second-price auction, the winner pays the second-highest bid (plus $0.01). This is incentive-compatible — advertisers' dominant strategy is to bid their true value.

```
Advertiser A: bid=$5, pCTR=0.02 → eCPM = $5 × 0.02 × 1000 = $100
Advertiser B: bid=$8, pCTR=0.01 → eCPM = $8 × 0.01 × 1000 = $80
Advertiser C: bid=$3, pCTR=0.04 → eCPM = $3 × 0.04 × 1000 = $120  ← WINS

Winner C pays: eCPM_second / pCTR_winner = $100 / 0.04 = $2.50 per click
(Not their bid of $3 — they pay based on the second-highest eCPM)
```

**Quality score** penalizes low-quality ads (poor landing pages, low engagement history):

$$\text{eCPM} = \text{bid} \times \hat{p}_{CTR} \times \text{quality\_score}$$

### Position Bias Correction

Users click more on position 1 than position 5 regardless of ad quality. CTR data conflates ad quality with position effect.

**Examination hypothesis:** 
$$P(\text{click}) = P(\text{examine position } k) \times P(\text{click} \mid \text{examined})$$

Training without position correction causes the model to learn that ads historically shown in position 1 have high CTR — a feedback loop.

**Solutions:**

1. **Propensity scoring:** Weight training examples by inverse propensity of their shown position
2. **Position as a feature:** Include position_rank as a feature at training time; set position=1 at inference (we decide position after scoring)
3. **Paired training:** For each impression, generate synthetic examples at all positions with appropriate CTR estimates

```python
def position_debias_weight(position: int, position_propensities: dict) -> float:
    """
    Inverse propensity score (IPS) weighting.
    position_propensities[k] = P(shown at position k | selected)
    """
    return 1.0 / position_propensities.get(position, 1.0)

# Typical propensities (estimated from randomization experiments):
# position 1: 0.40, position 2: 0.25, position 3: 0.15, ...
```

### Counterfactual Evaluation

Because the training data only contains ads that were selected by the previous model, offline evaluation is biased. Counterfactual evaluation estimates what would have happened under a new policy using logged data.

**Inverse Propensity Scoring (IPS) for offline eval:**

$$\hat{V}_{IPS}(\pi_{new}) = \frac{1}{n} \sum_{i=1}^n \frac{\pi_{new}(a_i | x_i)}{\pi_{old}(a_i | x_i)} \cdot r_i$$

High-variance for rare actions (rare ad selections). Doubly-robust estimators (DM + IPS) reduce variance.

---

## 9. A/B Testing Challenges

### Interference Effects (SUTVA Violations)

The Stable Unit Treatment Value Assumption (SUTVA) requires that a user's outcome depends only on their own treatment, not others'. Ads violate this:

- Advertiser A is in experiment group; their ads now score higher → they win more auctions → Advertiser B's ads (in control group) get fewer impressions
- Measuring "CTR improvement" in treatment vs control is confounded by budget reallocation
- An experiment that improves CTR by 0.5% might decrease control group CTR by 0.3% — net lift is only 0.2%, but naive experiment reports 0.5%

**Mitigations:**

| Approach | Mechanism | Trade-off |
|---|---|---|
| Geo-based experiments | Assign entire geographies to treatment/control | Reduces interference; requires many geos for power |
| Advertiser-based holdout | Split advertisers rather than users | Clean for advertiser-level metrics; can't test user experience |
| Budget-constrained experiments | Hold advertiser budgets constant across arms | Reduces spillover; harder to operationalize |
| Switchback experiments | Alternate treatment/control in time slots (e.g., hour-by-hour) | Works for time-stationary effects; problematic if effects persist |

### Metrics in Ads Experiments

```
Primary guardrail metrics (must not regress):
  - Revenue per query (RPQ)
  - Advertiser spend rate (budget utilization)
  - User satisfaction signals (skip rate, complaint rate)

Primary success metrics:
  - CTR lift (with confidence intervals)
  - RPM improvement
  - Advertiser ROI

Secondary metrics (directional):
  - Model log loss improvement
  - Calibration ECE
  - Impression share for small advertisers
```

### Sample Size and Minimum Detectable Effect

CTR experiments require large sample sizes due to high variance:

```python
def required_sample_size(baseline_ctr=0.02, mde=0.001, alpha=0.05, power=0.80):
    """
    MDE = minimum detectable effect (absolute CTR change)
    At 2% baseline CTR, detecting 0.1% absolute change needs ~3M impressions per arm.
    """
    from scipy.stats import norm
    z_alpha = norm.ppf(1 - alpha/2)
    z_beta = norm.ppf(power)
    p = baseline_ctr
    delta = mde
    n = (z_alpha + z_beta)**2 * 2 * p * (1-p) / delta**2
    return int(n)

# required_sample_size(0.02, 0.001) ≈ 3,000,000 impressions per arm
# At 10M impressions/second, this takes <1 second in production!
# But require 2-week minimum for day-of-week effects to average out.
```

---

## 10. Failure Modes

### Feedback Loop: The Matthew Effect

**Problem:** Ads with high predicted CTR are shown more often → collect more clicks → model assigns them even higher pCTR. New ads with no history are never shown, never learn a good CTR estimate.

```
pCTR(ad_A) high → wins auction → shown more → more clicks observed
                                                         │
                                              ▼
                                   retrain → pCTR(ad_A) even higher
```

**Mitigations:**
- **Exploration budget:** Reserve ε fraction of auctions for random or UCB-based ad selection (analogous to multi-armed bandit exploration)
- **Thompson sampling:** Sample pCTR from posterior distribution rather than using point estimate — naturally balances explore/exploit
- **Counterfactual off-policy training:** Train on all candidate ads, not just winners (requires logging propensities)

### Cold Start for New Ads

New ad has zero impressions → no click history → embedding initialized randomly → model has no signal.

**Solutions:**

| Strategy | How |
|---|---|
| Content-based initialization | Initialize ad embedding from creative embedding (image/text features) |
| Advertiser transfer | Initialize from average embedding of same advertiser's other ads |
| Category prior | Initialize from category-level CTR prior |
| Bayesian prior | Start with smoothed CTR = global_avg_ctr × (1 + pseudo_count) / (impressions + pseudo_count) |

```python
def cold_start_ctr_estimate(impressions: int, clicks: int,
                             category_ctr: float, pseudo_count: float = 100.0) -> float:
    """
    Bayesian smoothing toward category prior.
    With 0 impressions: returns category_ctr.
    With 10K impressions: nearly pure empirical CTR.
    """
    prior_clicks = category_ctr * pseudo_count
    posterior_ctr = (clicks + prior_clicks) / (impressions + pseudo_count)
    return posterior_ctr
```

### Cold Start for New Users

New user visits the site for the first time.

**Solutions:**
- **Context-only serving:** Use device type, geo, time-of-day, page content without user history
- **Demographic fallback:** Age + gender (if provided) → use segment-level CTR
- **Session signals:** Clicks within current session bootstrap a session embedding in real time
- **Federated/ondevice signals:** In privacy-preserving settings, use on-device behavioral signals without sending to server

### Distribution Shift

**Holiday seasons:** Black Friday CTR is 3–5× normal. A model trained on October data drastically underestimates November demand.

**Detection:**

```python
def detect_distribution_shift(reference_scores, current_scores,
                               threshold_psi=0.2) -> bool:
    """
    Population Stability Index (PSI) — standard in ad/credit industries.
    PSI < 0.1: no shift
    PSI 0.1–0.2: moderate shift, monitor
    PSI > 0.2: significant shift, retrain
    """
    bins = np.percentile(reference_scores, np.linspace(0, 100, 11))
    ref_dist = np.histogram(reference_scores, bins=bins)[0] / len(reference_scores)
    cur_dist = np.histogram(current_scores, bins=bins)[0] / len(current_scores)
    
    ref_dist = np.clip(ref_dist, 1e-6, None)
    cur_dist = np.clip(cur_dist, 1e-6, None)
    
    psi = np.sum((cur_dist - ref_dist) * np.log(cur_dist / ref_dist))
    return psi > threshold_psi
```

**Mitigations:**
- Shorter training window (last 7 days instead of 30 days) for faster adaptation
- Sample-weighted training: weight recent examples more heavily
- Pre-season fine-tuning: collect holiday traffic data from prior year, fine-tune before the event

### Click Fraud

Invalid clicks inflate CTR, poison training labels, waste advertiser budgets.

**Detection signals:**
- Click velocity: >10 clicks/minute from same IP
- Click-through without dwell time: user clicks, immediately leaves (<2 seconds on landing page)
- Geographic anomalies: IP claims to be in New York, but timezone/language is Eastern Europe
- Click farms: coordinated click patterns across IPs sharing subnet

**System response:**
- Invalid click filtering before labels enter training data
- Separate "raw CTR" vs "valid CTR" metrics
- Retroactive budget credits to affected advertisers

---

## 11. Interview Angles

### Q1: Why is log loss a better training objective than AUC for CTR models? [Medium]

AUC is not differentiable and measures ranking, not calibration. Log loss directly minimizes the negative log-likelihood of the click labels — it penalizes confidently wrong predictions (predicting pCTR=0.9 when the user didn't click costs much more than predicting 0.6). In the auction, we need calibrated absolute probabilities, not just relative ranking. A model with perfect AUC but systematic 2× overestimation of pCTR would cause every advertiser to overpay. Log loss optimizes for both ranking and calibration simultaneously.

---

**Cross-questions to expect:**
- *"Is log loss actually optimizing calibration, or does it just happen to?"* → It's a strictly proper scoring rule: its expected value is uniquely minimized by the true conditional probability. Calibration isn't a side effect, it's what properness means. Brier score is also proper — log loss is preferred because it penalizes confident errors much harder, which is what you want when the output feeds a price.
- *"If it's proper, why do you still need a Platt/isotonic layer on top?"* → Because properness holds at the optimum over an unrestricted hypothesis class. A real model is misspecified, early-stopped, regularized, and trained on negatively-sampled data — all of which break the guarantee. The post-hoc layer repairs the gap you actually have, not the one in theory.
- *"You negatively sample. What does that do to the output?"* → Biases it upward by a known factor. Sampling negatives at rate $r$ gives $p_{\text{obs}} = p/(p + (1-p)r)$; you invert it to recover the true scale. Forgetting this recalibration is one of the most common sources of a systematic overbid.

**Trap:** Saying "AUC isn't differentiable" as the whole answer. There are differentiable ranking surrogates, so that objection alone is weak. The real reason is that the auction consumes an absolute probability — a rank is not a price.

---
### Q2: How do you handle the fact that embedding tables for user IDs don't fit on a single GPU? [Hard]

Meta's DLRM paper describes the solution: model parallelism for embedding tables, data parallelism for dense layers. Each GPU (or host) owns a shard of each embedding table. When a forward pass needs embedding lookups, an all-to-all communication collective sends each lookup request to the shard that owns it and returns the embedding vector. Dense MLP layers are replicated on all workers and synchronized via all-reduce (standard DDP). The communication overhead is the main scaling bottleneck — minimizing embedding dimension and table size is critical.

---

**Cross-questions to expect:**
- *"All-to-all is the bottleneck — how do you shrink it?"* → Shard by table for small tables and row-wise for large ones so load stays even; fuse many lookups into one collective; keep embedding dimensions small (8–32 is common for tail features, not 128). Frequency-based hashing so rare IDs share rows cuts table size at a modest accuracy cost.
- *"Do you need a row per user ID at all?"* → Usually not. Most user IDs appear a handful of times and their embeddings never leave the initialization neighbourhood. Hashing the tail into a shared bucket, or dropping IDs below a frequency floor and relying on aggregate features, costs little and saves most of the table.
- *"Why is a parameter server acceptable here when DDP is standard elsewhere?"* → Because embedding gradients are extremely sparse — a batch touches a tiny fraction of rows, so all-reduce over the full table is enormous waste. Async parameter-server updates suit sparse access; the staleness that would hurt dense training is tolerable for rarely-touched rows.

**Trap:** Describing this as "just model parallelism." The split is the point: model-parallel embeddings, data-parallel dense layers, in one model. Candidates who apply a single strategy to both halves miss the design.

---
### Q3: You ship a new CTR model and revenue drops 3% despite the offline AUC improving by 0.002. What happened? [Hard]

Classic calibration regression. The new model has better ranking (AUC) but worse calibration — it systematically underestimates or overestimates pCTR. Even if it ranks ad A above ad B correctly, if the predicted probabilities are too low, the eCPMs computed in the auction are too low, causing the platform to charge advertisers less per click (revenue drop). Always check ECE (Expected Calibration Error) in offline evaluation, not just AUC. Also check: was negative sampling rate changed? Was the post-hoc calibration layer (isotonic regression / Platt) retrained on the new model's outputs?

---

**Cross-questions to expect:**
- *"Name a cause other than calibration."* → An auction-dynamics shift. Better ranking reallocates impressions toward ads with higher pCTR but lower bids, so eCPM per impression falls even with correct calibration. Revenue depends on the bid–pCTR joint distribution, not pCTR quality alone.
- *"How would you distinguish those two in an hour?"* → Plot a reliability diagram on the holdout for both models. Miscalibration shows as a curve off the diagonal; if both are calibrated, the cause is allocation, and you check the eCPM and bid distribution of the newly-winning ads.
- *"Is 0.002 AUC even a real improvement?"* → Probably not on its own. On billions of rows it may be statistically significant and still practically meaningless — well inside the variance from seed and data window. Treat it as noise unless it reproduces across reseeds.

**Trap:** Rolling back and stopping. The rollback is right; the finding is that your offline gate lets a revenue regression through. If ECE isn't a blocking check alongside AUC, the same failure ships again next quarter.

---
### Q4: Explain position bias and how you correct for it in training. [Medium]

Position bias is the tendency for users to click more on higher-positioned ads regardless of ad quality, because they are more likely to look at them (the "examination hypothesis"). If we train without correction, the model learns that ad IDs historically served in position 1 have high CTR — because they were examined more, not because they're better. This creates a feedback loop where ads that happened to win early auctions dominate forever. Correction approaches: (1) include position as a training feature and set position=1 at inference time; (2) inverse propensity score weighting where examples from lower positions are upweighted by the inverse of the position's click probability; (3) dueling bandits or randomized experiments to estimate position propensities.

---

**Cross-questions to expect:**
- *"Where do the propensities come from?"* → Ideally from deliberate randomization — swapping positions on a small traffic slice gives an unbiased estimate. Without that, intervention harvesting mines naturally-occurring position changes for the same ad, or you estimate position and relevance jointly with an EM-style model. Assuming a decay curve is the weakest option.
- *"What does IPS cost you?"* → Variance. Low-position examples get large weights, so a few rare events dominate the gradient. Clipping the weights trades a little bias for a large variance reduction and is standard practice.
- *"Position-as-a-feature is simpler — why not just do that?"* → It only works if position is conditionally independent of quality given your features, which is false when the previous ranker placed better ads higher. The feature then absorbs relevance signal, and forcing position=1 at inference extrapolates off-distribution.

**Trap:** Treating this as a ranking-metrics nuisance. It's a feedback loop: uncorrected, today's winners are tomorrow's training signal, and the system slowly stops being able to discover anything new.

---
### Q5: Why can't you just use a standard train/validation split for offline evaluation of a CTR model? [Medium]

Two problems. First, **temporal leakage**: if you randomly split data, you're evaluating on examples that are earlier in time than some training examples. The model has "seen the future." Always use a temporal split: train on days 1–29, validate on day 30. Second, **survivorship bias / policy bias**: the validation set only contains ads that were shown by the previous model. If the new model would show a different set of ads, you have no data for those impressions — you cannot measure the new model's performance on counterfactual decisions. This requires counterfactual evaluation (IPS) or online A/B testing to resolve.

---

**Cross-questions to expect:**
- *"Temporal split fixes leakage — what does it break?"* → Your validation set is now one specific day, so it carries that day's idiosyncrasies: a holiday, an outage, a campaign launch. Use several consecutive holdout days, or rolling-origin evaluation, before believing a small delta.
- *"How do you evaluate counterfactually without shipping?"* → IPS or doubly-robust estimators over logged propensities, which requires you to have logged the serving policy's probabilities. Most teams discover too late that they never logged them — the fix is architectural and has to precede the need.
- *"So why not skip offline evaluation and A/B everything?"* → Traffic is the scarce resource. Offline evaluation is a filter that decides what deserves an experiment slot; it is not a substitute for one.

**Trap:** Claiming a temporal split makes the estimate unbiased. It removes leakage, not policy bias — the holdout still only contains ads the old model chose to show.

---
### Q6: How would you design the system to handle 100B training examples per day? [Hard]

At 100B examples/day ≈ 1.16M examples/second, single-machine training is impossible. Key decisions: (1) **Data pipeline**: stream examples through Kafka → Flink for feature joins → write to distributed storage (HDFS/S3) as Parquet shards; (2) **Training**: PyTorch DDP for dense layers across 100+ GPUs; parameter server or model-parallel sharding for embedding tables; (3) **Mini-batch SGD** with ~4K–64K batch size per step; (4) **Continuous training**: rather than full retraining daily, continuously fine-tune on recent data with a sliding window; new model checkpoint every 15–60 minutes; (5) **Sparse updates**: use EmbeddingBag with `sparse=True` to update only the embeddings that appeared in each batch — crucial for efficiency when vocabulary is billions of IDs but each batch only touches thousands.

---

**Cross-questions to expect:**
- *"Do you need all 100B?"* → Almost never. Negatives massively outnumber positives; sampling negatives at 1–10% and recalibrating retains nearly all the signal at a fraction of the cost. Say this first — a candidate who scales the cluster before questioning the data volume is solving the expensive version of the problem.
- *"One pass or many epochs?"* → One. At this volume, fresh data beats re-reading old data, and single-pass streaming training avoids the memorization that multi-epoch training invites on high-cardinality IDs.
- *"What breaks with a 64K batch?"* → The effective learning rate. You need LR scaling with warmup, and past a critical batch size the returns flatten — you're buying throughput, not convergence. Very large batches also smooth away the rare-event gradients that sparse features depend on.

**Trap:** Answering only with infrastructure. The interviewer is usually probing whether you know that continuous incremental training, not nightly full retraining, is what the freshness requirement actually demands.

---
### Q7: An advertiser complains their ads stopped getting impressions after your new CTR model deployed. How do you investigate? [Medium]

Structured debugging checklist:
1. **Check pCTR trend for their ads**: Did the new model assign systematically lower pCTR to their ads vs old model? Pull impression logs and compare pCTR distributions before/after model swap.
2. **Check eCPM competitiveness**: Even if pCTR is similar, their eCPM might now fall below auction floor due to model calibration change. Compare eCPM percentile rank for their ads.
3. **Check for cold start regression**: Did the new model change the embedding initialization strategy? Their ads might be newly classified as "new" by the model.
4. **Check feature coverage**: Are all features for their ads present in the feature store? A missing feature defaults to zero/null and can tank predictions.
5. **Check for category-level calibration bug**: If their ads belong to a specific category where the model is miscalibrated, run category-level calibration curves.
6. **Compare model scores on held-out examples for their specific ads**: Use shadow mode (run old and new model in parallel) to compare score distributions.
7. **Check for data leakage in training**: If their ads' recent impressions were excluded from the training window by accident, the model would have no signal for them.

---
**Cross-questions to expect:**
- *"What do you check before touching the model?"* → Whether anything changed on their side — budget exhausted, pacing, bid lowered, targeting narrowed, creative rejected. A model deploy is a salient coincidence, and advertiser-side changes are the more common cause. Confirm the correlation is real before investigating it.
- *"Suppose the model genuinely deprioritized them and is correctly calibrated. Is that a bug?"* → No — that's the system working. The response is commercial, not technical. Knowing which findings need a fix and which need an account conversation is part of the answer.
- *"How do you get an answer without a rollback?"* → Shadow scoring. You are already logging both models' scores on live traffic, so you can compare pCTR distributions for their ad IDs directly. If you aren't, that's the actual gap the incident exposed.

**Trap:** Diving into feature-store debugging first. Start by segmenting: is this advertiser, their vertical, or everyone? A single complaint is a sample of one, and the segment size determines whether you're chasing a bug or a rounding error.

---

## Appendix: Key Papers and Systems

| Paper / System | Contribution | Year |
|---|---|---|
| McMahan et al. — Ad Click Prediction at Google (FTRL) | Online FTRL-Proximal for sparse LR; still a baseline | 2013 |
| Cheng et al. — Wide & Deep Learning (Google) | Wide (memorization) + Deep (generalization) joint model | 2016 |
| Guo et al. — DeepFM (Huawei) | FM replaces manual wide features; end-to-end | 2017 |
| Wang et al. — DCN v2 (Google) | Explicit polynomial feature crossing at scale | 2021 |
| Naumov et al. — DLRM (Meta) | Efficient architecture separating sparse/dense; open-sourced | 2019 |
| He et al. — Practical Lessons from Predicting Clicks on Ads at Facebook | GBDT + LR pipeline; field normalization; importance of online learning | 2014 |
| Chapelle — Modeling Delayed Feedback in Display Advertising | Survival analysis for conversion attribution | 2014 |
| Criteo Display Advertising Dataset | Public benchmark; 45M examples, 26 categorical features | 2014 |

---

## Appendix: Scale Cheat Sheet

```
Impressions:         10M/second  → 864B/day
Training examples:   100B/day    → ~1.2M/second
P99 scoring latency: <10ms       → 2ms feature fetch + 8ms GPU inference
Embedding tables:    192GB+      → must shard across CPU memory / hosts
Model update cadence: 15–60 min  → continuous training, not daily batch
Negative sampling:   1:100 ratio → correct calibration bias at inference
Calibration target:  ECE < 0.002
Log loss improvement threshold: 0.001 → production-worthy at Google scale
```

## Flashcards

**Why does calibration matter more than ranking quality (AUC) for CTR models?** #flashcard
The auction uses the absolute value of pCTR to compute eCPM (bid × pCTR × quality); if pCTR is systematically 2× too high, every advertiser overpays by 2× regardless of how well the model ranks ads relative to each other.

**Why is feature hashing necessary for user×ad cross features instead of storing them explicitly?** #flashcard
The cross-feature space is combinatorially too large (trillions of possible pairs) to store exactly; hashing projects pairs into a fixed-size space (e.g. 2^26) with a small, acceptable collision rate (~0.015%).

**Why does DLRM separate sparse (ID) and dense features before combining them, rather than concatenating everything upfront?** #flashcard
Sparse IDs need embedding-table lookups and benefit from explicit pairwise interaction (dot products) with each other; dense features go through a bottom MLP first. Treating them uniformly would waste the structure of the ID space and blow up parameter count.

**Why use a two-tower architecture for retrieval but a single joint model for ranking?** #flashcard
Retrieval must search ~1B candidates in milliseconds, so user and ad embeddings are computed independently and matched via fast ANN search (dot product); ranking only scores ~1000 candidates, so a joint model can afford full user-ad feature interactions for higher accuracy.

**Why must you apply a calibration correction after negative sampling, and what happens if you forget?** #flashcard
Training on a downsampled negative class shifts the learned probability distribution; without correcting via `p_true = p_model / (p_model + (1-p_model)/q)`, the raw model output is inflated (roughly 100× too high at a 1% sampling rate), breaking the auction's eCPM calculation.

**Why does training a CTR model without position-bias correction create a feedback loop?** #flashcard
Users click higher-positioned ads more regardless of quality (the examination hypothesis); an uncorrected model learns "shown in position 1 → high CTR" and keeps ranking those same ads to position 1, starving new/lower-ranked ads of the exposure needed to prove themselves.

**Why can't a standard random train/validation split be used to evaluate a CTR model offline?** #flashcard
Two reasons: temporal leakage (random splits let the model train on future information relative to some validation examples — use a strict time-based split), and survivorship bias (the logged data only contains ads the previous policy chose to show, so a new policy's counterfactual choices have no ground truth — requires IPS/counterfactual evaluation).

**Why do CTR model experiments face interference effects that break standard A/B test assumptions (SUTVA)?** #flashcard
Advertisers share a fixed budget/auction pool; giving the treatment group's ads better scores causes them to win more auctions, directly reducing the control group's impressions — the groups aren't independent, so naive lift measurement overstates the true effect.

**Why is the "Matthew effect" a failure mode specific to CTR feedback loops, and how is it mitigated?** #flashcard
Ads with high predicted CTR get shown more, accumulate more clicks, and are retrained to even higher pCTR, while new ads never get exposure to prove themselves; mitigated with exploration budgets, Thompson sampling (sampling from the posterior instead of a point estimate), or counterfactual off-policy training on all candidates, not just past winners.

**Why does Population Stability Index (PSI) matter for CTR models specifically around holidays?** #flashcard
Events like Black Friday shift CTR 3-5× above normal; a model trained on pre-holiday data will systematically underestimate pCTR during the shift. PSI compares score distributions to flag when retraining or a shorter/reweighted training window is needed.
