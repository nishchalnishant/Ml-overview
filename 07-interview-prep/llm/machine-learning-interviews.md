# Machine Learning Interview — Complete Preparation Guide

---

## 1. Interview Loop Structure

| Round | Focus | What you're actually being tested on |
| :--- | :--- | :--- |
| Recruiter screen | Background, motivation | Communication, role fit |
| DSA / coding | Arrays, graphs, DP | Problem decomposition speed |
| ML coding | Implement from scratch | Whether you can translate math to code |
| ML theory | Concepts, tradeoffs | Depth of understanding beyond definitions |
| ML system design | End-to-end system | Production thinking, prioritization |
| Behavioral / project deep dive | Past work | Judgment calls, ownership, impact |

**Calibrate prep time by level:**
- L3/junior: 70% theory + coding, 30% system design
- L4/mid: 50% system design, 30% theory, 20% behavioral
- L5+/senior: 60% system design + behavioral, 40% theory

---

## 2. The Question Shift by Level

**Junior:** "What is X?"
- Expected: correct definition + formula

**Mid:** "Why X and not Y in this situation?"
- Expected: tradeoffs table + context-dependent recommendation

**Senior:** "Design a system that uses X at scale. What breaks first?"
- Expected: end-to-end design + failure modes + monitoring strategy

---

## 3. ML Theory — Answering Framework

Every theory question has three layers. Hit all three:

```
Layer 1: Definition + formula
  → "Dropout randomly zeros activations with probability p during training,
     scaling surviving activations by 1/(1-p)."

Layer 2: Intuition
  → "This prevents co-adaptation — neurons can't rely on specific others,
     so each must learn independently. Acts like ensemble averaging."

Layer 3: Tradeoff / when NOT to use
  → "Hurts smaller models (too much information lost). Don't use with
     BatchNorm — their interaction causes training instability."
```

**Common theory questions and what they're really testing:**

| Question | They're testing |
| :--- | :--- |
| Bias-variance tradeoff | Whether you think about generalization, not just accuracy |
| Why ReLU? | Understanding gradient flow and vanishing gradients |
| BatchNorm vs LayerNorm | Transformer vs CNN knowledge, small-batch awareness |
| L1 vs L2 regularization | Probabilistic interpretation (MAP → Lasso/Ridge) |
| Why cross-entropy over MSE for classification? | Log-likelihood, gradient magnitudes near decision boundary |
| Precision vs recall tradeoff | Business context — cost of FP vs FN |

---

## 4. ML Coding — What You Must Write Without Hesitation

### 4.1 The 6 Must-Know Implementations

**1. Sigmoid + BCE loss**
```python
import numpy as np

def sigmoid(z): return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def bce_loss(y_true, y_pred, eps=1e-7):
    return -np.mean(y_true * np.log(y_pred + eps) + (1 - y_true) * np.log(1 - y_pred + eps))
```

**2. Softmax (numerically stable)**
```python
def softmax(x):
    x = x - x.max(axis=-1, keepdims=True)  # subtract max: prevents exp() overflow
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)
```

**3. Precision, Recall, F1**
```python
def prf1(y_true, y_pred, eps=1e-8):
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    p = tp / (tp + fp + eps)
    r = tp / (tp + fn + eps)
    return p, r, 2 * p * r / (p + r + eps)
```

**4. Scaled dot-product attention**
```python
def attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(0, 2, 1) / np.sqrt(d_k)
    if mask is not None:
        scores = np.where(mask, scores, -1e9)
    weights = softmax(scores)
    return weights @ V
```

**5. K-Means (one iteration)**
```python
def kmeans_step(X, centroids):
    dists = np.linalg.norm(X[:, None] - centroids[None], axis=2)  # (n, k)
    labels = dists.argmin(axis=1)
    new_centroids = np.array([X[labels == j].mean(axis=0) for j in range(len(centroids))])
    return labels, new_centroids
```

**6. Simple gradient descent step**
```python
def gd_step(X, y, w, b, lr=0.01):
    n = len(y)
    y_hat = sigmoid(X @ w + b)
    dw = (X.T @ (y_hat - y)) / n
    db = (y_hat - y).mean()
    return w - lr * dw, b - lr * db
```

### 4.2 PyTorch Patterns (Must Know Cold)

```python
# Training loop essentials
optimizer.zero_grad()          # always before backward
loss.backward()
nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # prevent exploding gradients
optimizer.step()
scheduler.step()

# Inference
model.eval()
with torch.no_grad():
    out = model(x)

# Save / load checkpoint
torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, "ckpt.pt")
ckpt = torch.load("ckpt.pt", map_location=device)
model.load_state_dict(ckpt["model"])
```

---

## 5. ML System Design — Universal Structure

Use this sequence for every problem. Never skip steps:

```
Step 1: Clarify goal
  → "What user action are we optimizing? Click, purchase, dwell time?"
  → "Is this a ranking, classification, or generation problem?"

Step 2: Define metrics
  → Online:  CTR, revenue/query, D7 retention, latency P99
  → Offline: AUC-ROC, NDCG@10, precision@K, perplexity
  → "What does success look like in a business dashboard 30 days post-launch?"

Step 3: Constraints
  → Latency SLA (< 100ms? < 2s?), QPS, memory budget, regulatory requirements

Step 4: Data and labels
  → Volume? Freshness needed? Label quality? Weak vs strong supervision?
  → Training-serving skew risks?

Step 5: Baseline
  → Start with the simplest thing: popularity ranking, LR, BM25
  → This is your control in the A/B test

Step 6: Model architecture
  → Justified by constraints, not trend
  → Two-stage if scale > 1M items: retrieval (ANN/BM25) → ranking (GBT/DNN)

Step 7: Feature pipeline
  → Real-time (Redis velocity features) vs batch (offline embeddings)
  → Where are transforms fit? (train only — no leakage)

Step 8: Serving
  → Batch (offline pre-computation) vs real-time (< 200ms)
  → Caching strategy for expensive lookups

Step 9: Evaluation
  → A/B test design: metric, power, duration, novelty effect mitigation
  → Holdout set, backtesting for ranking

Step 10: Monitoring
  → PSI on features (> 0.25 = retrain), prediction score distribution
  → Business KPI drop threshold for auto-rollback
```

**Senior-level signals** (always mention these):

| Signal | Why it matters |
| :--- | :--- |
| Cold start | Every real system has new users / new items |
| Feedback loops | Popularity → more data → more popular (self-reinforcing bias) |
| Rollback plan | Shows production maturity |
| Cost-aware threshold | Precision vs recall is a business cost question |
| Data freshness | Stale features cause silent degradation |

---

## 6. Behavioral / Project Deep Dive — Structure

**Every project story needs all 7 elements:**

| Element | Bad version | Strong version |
| :--- | :--- | :--- |
| Problem | "We needed better recommendations" | "CTR on homepage dropped 8% after a catalog expansion — existing model had no cold start handling for new items" |
| Constraints | (omitted) | "< 50ms latency, no model retraining more than weekly, team of 2" |
| Approach | "We used two-tower model" | "We tried matrix factorization first (baseline), identified cold start as bottleneck, then added content-based two-tower for new items" |
| Tradeoffs | (omitted) | "Two-tower adds 40ms vs MF, justified because it recovered the 8% CTR on new items. Accepted worse recall@100 for existing items" |
| Metrics | "AUC improved" | "Offline NDCG@10 +12%. A/B test: CTR +6.3%, revenue/user +2.1%, p<0.01 at n=500K per arm" |
| Production outcome | "We shipped it" | "Rolled out with 5% canary → 50% → 100% over 2 weeks, monitoring PSI and CTR. Zero rollbacks needed" |
| Lesson | (omitted) | "Diagnosed cold start too late — should've been in the baseline evaluation. Now cold start is part of every evaluation checklist" |

**The three story types you must have ready:**

1. **Technical problem-solving story:** diagnosed a hard bug or unexpected degradation
2. **Failure story:** something didn't work and you shipped it anyway, or got it wrong first
3. **Tradeoff / disagreement story:** chose the slower but more reliable approach over a stakeholder's preference

---

## 7. Key Tradeoffs — What "Why X and Not Y" Sounds Like

| Tradeoff | Strong answer structure |
| :--- | :--- |
| XGBoost vs neural net | "XGBoost: tabular data with < 1M rows, faster to iterate, interpretable. Neural net: when you need to jointly learn embeddings (IDs, text) or when you have > 10M rows. Both: A/B test wins." |
| Precision vs recall | "Depends on cost. Fraud: FN (missed fraud) >> FP (false block). Cancer screening: same. Spam filter: FP (blocking legit email) >> FN. Cost matrix → threshold." |
| BatchNorm vs LayerNorm | "BatchNorm normalizes over the batch — breaks for small batches and variable-length sequences. LayerNorm normalizes over features per sample — standard for Transformers, works at batch size 1." |
| L1 vs L2 regularization | "L1: sparse weights (Lasso) — useful for feature selection, interpretability. L2: small weights, none exactly zero (Ridge) — better when all features are relevant. L1 = Laplace prior, L2 = Gaussian prior on MAP." |
| Online vs batch serving | "Online: decision depends on current context (fraud at transaction time). Batch: when freshness < latency requirement matters (weekly churn scores). Hybrid: batch pre-compute user embeddings, online lookup." |
| SMOTE vs class_weight | "class_weight='balanced' is free and works for most models. SMOTE generates synthetic minority samples — helps when minority class is severely underrepresented in tabular data. Doesn't help with distribution shift." |

---

## 8. Numbers You Should Know Cold

| Fact | Value |
| :--- | :--- |
| GPT-3 parameters | 175B |
| Chinchilla optimal tokens/param | ~20 |
| LLaMA 3 training tokens | 15T |
| Standard attention complexity | $O(n^2 d)$ |
| LoRA trainable params (typical) | ~0.06% of base |
| BF16 memory/param | 2 bytes |
| 70B model BF16 VRAM | ~140GB |
| PSI > X = retrain trigger | 0.25 |
| KS test p-value threshold | 0.05 |
| AdamW typical β₁, β₂ | 0.9, 0.999 |
| Dropout rate range | 0.1–0.5 |
| Warmup steps (typical) | 1–5% of total |

---

## 9. Mistakes to Avoid

| Mistake | Why it costs you |
| :--- | :--- |
| Starting design with model choice | Shows no requirements thinking |
| Only mentioning accuracy as metric | No business metric awareness |
| Skipping cold start in recommendations | Junior signal |
| "Retrain the model" as first response to prod issue | Shows no debugging instinct |
| Not mentioning monitoring or rollback | Shows no production experience |
| Overconfident about tricky stats questions | Misinterpreting p-value is a red flag |
| Rote answers to behavioral questions | No ownership or judgment shown |

---

## 10. Prep Checklist

**Must be cold (0 hesitation):**
- [ ] Sigmoid, softmax, BCE from scratch
- [ ] Precision, recall, F1 from scratch
- [ ] Bias-variance decomposition
- [ ] Bayes theorem calculation
- [ ] p-value correct interpretation (and 2 wrong interpretations)
- [ ] Two-stage retrieval+ranking pattern
- [ ] 10-step system design framework
- [ ] Three project stories (solve, failure, disagreement)

**Must be able to explain tradeoffs:**
- [ ] L1 vs L2 vs Dropout vs Early stopping
- [ ] BatchNorm vs LayerNorm
- [ ] Batch vs real-time serving
- [ ] Precision vs recall (with cost framing)
- [ ] XGBoost vs neural net

**Must be able to sketch from memory:**
- [ ] Two-tower recommendation architecture
- [ ] Fraud detection pipeline (rule engine + ML + threshold)
- [ ] LLM serving stack (load balancer → vLLM → KV cache)
- [ ] Train/val/test split (including temporal split for time series)
