---
module: Production Ml
topic: System Design
subtopic: Fraud Detection Full System
status: unread
tags: [productionml, ml, system-design-fraud-detection-]
---
# Fraud Detection System Design

End-to-end ML system for detecting fraudulent transactions in real time. Canonical system design question at fintech and payment companies (Stripe, PayPal, Meta Payments).

**Scale:** 10K TPS, <100ms decision latency, <0.1% false positive rate, $1B+ fraud prevented annually.

---

## 1. Problem Framing

### Clarifying Questions
- **What type of fraud?** Card-not-present, account takeover, synthetic identity, merchant fraud
- **Decision latency?** Real-time (<100ms) or near-real-time (<5s)
- **False positive cost?** Declined legitimate transaction → customer friction, churn
- **False negative cost?** Fraud loss + chargeback fee ($15–50/incident)
- **Feedback delay?** Chargebacks arrive 30–90 days after transaction

### Metric Trade-offs

**The precision-recall trade-off is the central problem:**

| Threshold | Precision | Recall | FPR | Business impact |
|---|---|---|---|---|
| Very strict | 99% | 30% | 0.01% | Miss 70% of fraud |
| Balanced | 85% | 75% | 0.15% | Balanced |
| Very aggressive | 50% | 95% | 0.5% | 0.5% good customers declined |

**Cost-sensitive threshold:**
$$\text{optimal threshold} = \arg\max_t \left[\text{Recall}(t) \cdot V_{fraud} - \text{FPR}(t) \cdot V_{FP}\right]$$

where V_fraud = average fraud value saved, V_FP = cost of declining a good transaction.

---

## 2. System Architecture

```
Transaction Request
        │
        ▼
 ┌──────────────┐      Feature Store
 │  Real-time   │◄────  (Redis, <1ms)
 │  Feature     │
 │  Enrichment  │◄────  External APIs
 └──────┬───────┘       (device, IP, velocity)
        │
        ▼
 ┌──────────────┐
 │  Rule Engine │──── Hard rules (block stolen cards, blocked countries)
 │  (fast path) │      <10ms
 └──────┬───────┘
        │ (if not blocked)
        ▼
 ┌──────────────┐
 │  ML Scoring  │──── Gradient boosted tree + neural embedding model
 │  (main path) │      30–50ms
 └──────┬───────┘
        │
        ▼
 ┌──────────────┐
 │  Decision    │──── APPROVE / DECLINE / REVIEW (3DS challenge)
 │  Engine      │
 └──────────────┘
        │
        ▼
 ┌──────────────┐
 │  Feedback    │──── Async: chargeback labels, manual review outcomes
 │  Loop        │      → retraining pipeline
 └──────────────┘
```

---

## 3. Feature Engineering

### Real-time Features (<1ms from Redis)

| Feature | Description | Signal |
|---|---|---|
| user_txn_count_1h | Transactions by user in last 1 hour | Velocity anomaly |
| user_spend_1h | Total spend in last 1 hour | Amount anomaly |
| merchant_fraud_rate_30d | Merchant's fraud rate last 30 days | Risky merchant |
| card_new_merchant_flag | Is this a new merchant for this card? | New payee risk |
| device_seen_before | Has this device been seen with this account? | Account takeover |
| ip_country_mismatch | Card country ≠ IP country | Geographic anomaly |
| time_since_last_txn | Seconds since previous transaction | Rapid succession |

### Batch Features (updated hourly/daily, served from Redis)

| Feature | Window | Update frequency |
|---|---|---|
| user_avg_txn_amount | 90 days | Daily |
| user_preferred_merchant_categories | 90 days | Daily |
| card_velocity_score | 7 days | Hourly |
| network_fraud_embedding | — | Weekly |

### Graph Features (velocity + network)

```python
def compute_velocity_features(user_id: str, redis_client, windows=[60, 3600, 86400]):
    """Compute transaction velocity in multiple time windows."""
    features = {}
    now = time.time()
    
    for window in windows:
        # Sorted set: score=timestamp, member=transaction_id
        count = redis_client.zcount(
            f"user:{user_id}:txns",
            min=now - window,
            max=now
        )
        features[f"txn_count_{window}s"] = count
    
    return features
```

### Feature Importance (typical GBM fraud model)

```
transaction_amount          0.18
user_txn_count_1h           0.14
time_since_last_txn         0.11
card_new_merchant_flag      0.09
ip_country_mismatch         0.08
merchant_fraud_rate_30d     0.07
device_seen_before          0.06
user_avg_txn_amount_90d     0.05
...
```

---

## 4. Model Architecture

### Two-Model Ensemble

**Model 1: Gradient Boosted Trees (GBT)**
- XGBoost or LightGBM
- Handles tabular features, missing values, non-linear interactions
- 500 trees, depth 6, ~30ms inference
- Interpretable via SHAP

**Model 2: Neural Embedding Model**
- Embed user history, merchant, device fingerprint
- Captures sequence patterns (series of transactions)
- ~20ms inference

**Ensemble:**
$$\text{fraud\_score} = 0.6 \cdot \text{score}_{GBT} + 0.4 \cdot \text{score}_{NN}$$

### GBT for Fraud (XGBoost)

```python
import xgboost as xgb
from sklearn.utils.class_weight import compute_sample_weight

# Class imbalance: ~0.1% fraud rate → weight positive class heavily
sample_weights = compute_sample_weight(
    class_weight={0: 1, 1: 1000},  # fraud is 1000x more costly
    y=y_train
)

model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=1000,  # alternative to sample weights
    eval_metric='aucpr',    # PR-AUC better than ROC for imbalanced
    early_stopping_rounds=50,
    tree_method='hist'      # fast histogram method
)

model.fit(
    X_train, y_train,
    sample_weight=sample_weights,
    eval_set=[(X_val, y_val)],
    verbose=100
)
```

### Sequence Model for Account Takeover

Account takeover (ATO) leaves a trail: new device login → password change → new payee added → large transfer. Sequential model captures this pattern.

```python
class TransactionSequenceModel(nn.Module):
    def __init__(self, feature_dim, hidden_dim=128, n_layers=2):
        super().__init__()
        # Embed each transaction
        self.feature_encoder = nn.Linear(feature_dim, hidden_dim)
        # LSTM over recent N transactions
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, n_layers, batch_first=True)
        # Predict fraud probability
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, transaction_history, current_txn):
        # transaction_history: [batch, seq_len, feature_dim]
        # current_txn: [batch, feature_dim]
        
        hist_encoded = self.feature_encoder(transaction_history)
        _, (h_n, _) = self.lstm(hist_encoded)
        context = h_n[-1]  # last layer hidden state
        
        # Concatenate context with current transaction
        current_encoded = self.feature_encoder(current_txn)
        combined = context + current_encoded
        return self.classifier(combined)
```

---

## 5. Handling Class Imbalance

**Fraud rate:** ~0.1% of transactions → 1000:1 imbalance.

**Strategies:**

| Strategy | How | When to use |
|---|---|---|
| `scale_pos_weight` | Weight positive class by 1/fraud_rate | GBM, always do this |
| SMOTE | Oversample minority class | Tabular, small datasets |
| Focal loss | Down-weight easy negatives | Neural models |
| Threshold tuning | Move decision boundary | Post-training |
| Undersampling | Reduce majority class | Very large datasets |

**Focal Loss** (originally for object detection, works for fraud):
$$\mathcal{L}_{focal} = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

where $(1-p_t)^\gamma$ down-weights easy negatives (high confidence correct predictions).

```python
def focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    p = torch.sigmoid(logits)
    p_t = targets * p + (1 - targets) * (1 - p)
    alpha_t = targets * alpha + (1 - targets) * (1 - alpha)
    focal_weight = alpha_t * (1 - p_t) ** gamma
    return (focal_weight * bce).mean()
```

---

## 6. Evaluation for Fraud

**Never use accuracy or ROC-AUC alone.**

**Primary metric: PR-AUC (Precision-Recall AUC)**
- Insensitive to true negatives (99.9% of transactions)
- Directly measures useful fraud detection at low FPR

**Operational metrics:**

```python
from sklearn.metrics import precision_recall_curve, average_precision_score

def fraud_evaluation_report(y_true, y_score, cost_per_fp=5, avg_fraud_value=200):
    """Comprehensive fraud model evaluation."""
    ap = average_precision_score(y_true, y_score)
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    
    print(f"PR-AUC: {ap:.4f}")
    
    # Find threshold at target FPR (e.g., 0.1%)
    from sklearn.metrics import roc_curve
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_score)
    target_fpr = 0.001
    idx = np.argmin(np.abs(fpr - target_fpr))
    operating_threshold = roc_thresholds[idx]
    
    y_pred = (y_score >= operating_threshold).astype(int)
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    
    fraud_saved = tp * avg_fraud_value
    fp_cost = fp * cost_per_fp
    net_value = fraud_saved - fp_cost
    
    print(f"At {target_fpr:.1%} FPR:")
    print(f"  Recall: {tp/(tp+fn):.2%}")
    print(f"  Fraud value saved: ${fraud_saved:,.0f}")
    print(f"  False positive cost: ${fp_cost:,.0f}")
    print(f"  Net value: ${net_value:,.0f}")
```

---

## 7. Feedback Loop and Label Delay

**The label delay problem:** Chargebacks arrive 30–90 days after the transaction. Training on only confirmed fraud misses "silent fraud" (unreported cases).

**Label hierarchy:**
```
Level 1 (immediate): Rule-based block (stolen card, blocked country)
Level 2 (days): Manual review outcome
Level 3 (weeks): Dispute filed by customer
Level 4 (months): Chargeback confirmed
```

**Training strategy:**
- Use Level 1+2 labels for daily model refresh (low delay, lower quality)
- Use Level 1–4 for monthly full retrain (high delay, high quality)
- Positive label = any level flagged, negative = 120 days without any flag

**Survival analysis for ambiguous labels:**
Transactions < 90 days old: use survival model to estimate P(fraud) given censored labels.

---

## 8. Concept Drift in Fraud

Fraud patterns evolve — new attack vectors appear, old ones get blocked.

**Temporal validation:**
```python
# Always use temporal split, never random split
train = df[df['txn_date'] < '2024-09-01']
val = df[(df['txn_date'] >= '2024-09-01') & (df['txn_date'] < '2024-10-01')]
test = df[df['txn_date'] >= '2024-10-01']

# Check for concept drift between train and test
from scipy.stats import ks_2samp
for col in features:
    stat, p = ks_2samp(train[col], test[col])
    if p < 0.01:
        print(f"Drift detected in {col}: KS={stat:.3f}")
```

**Retraining triggers:**
- Precision drops >5% at fixed threshold (new fraud pattern bypassing model)
- Recall drops >5% (false negatives increasing, unreported fraud rise)
- PSI > 0.25 on top-10 features (covariate shift)
- Fraud rate exceeds 2× expected (emerging attack)

---

## 9. Rules Engine + ML Hybrid

**Rule engine handles clear-cut cases fast:**
```python
class FraudRuleEngine:
    def evaluate(self, txn):
        # Hard blocks (immediate DECLINE)
        if txn['card_id'] in stolen_card_list:
            return 'DECLINE', 'stolen_card'
        if txn['merchant_country'] in blocked_countries:
            return 'DECLINE', 'blocked_country'
        if txn['amount'] > 50000 and txn['account_age_days'] < 7:
            return 'DECLINE', 'new_account_large_txn'
        
        # Soft flags (pass to ML with context)
        flags = []
        if txn['ip_country'] != txn['card_country']:
            flags.append('geo_mismatch')
        if txn['txn_count_1h'] > 10:
            flags.append('high_velocity')
        
        return 'REVIEW', flags  # ML model decides
```

**Rules vs ML trade-off:**
- Rules: interpretable, fast, easy to audit, brittle (adversaries adapt)
- ML: adaptive, catches subtle patterns, black box, needs labels

**Production:** Both in sequence. Rules for obvious fraud + regulatory compliance. ML for borderline cases.

---

## Canonical Interview Q&As

**Q: How do you handle the 30–90 day label delay in fraud model training?**  
A: Three approaches: (1) Tiered labels — use immediate signals (rule-based blocks, manual review) for frequent model updates, and delayed chargeback labels for periodic full retraining; (2) Censored label handling — transactions within 90 days without a chargeback are "censored" (potentially fraud not yet reported), treat them as unlabeled rather than negative; use survival analysis to estimate fraud probability; (3) Proxy labels — use dispute initiation (faster signal, ~7 days) as a leading indicator before chargeback confirmation. For drift detection without true labels: monitor prediction score distribution and feature PSI to catch new attack patterns before labels arrive.

**Q: Your fraud model has 98% accuracy. Is that good?**  
A: No — with 0.1% fraud rate, a model that always predicts "not fraud" gets 99.9% accuracy while catching zero fraud. Accuracy is useless for imbalanced fraud detection. The right metrics: PR-AUC (measures precision-recall trade-off), recall at fixed FPR (e.g., recall at 0.1% FPR), and dollar-weighted metrics (how much fraud value was prevented vs how many good customers were declined). The operational question is: at our target false positive rate, what fraction of fraud do we catch?

**Q: How would you design the feedback loop for a real-time fraud model?**  
A: (1) Log every decision with features at decision time; (2) Join chargeback and dispute labels back to original transaction records using transaction_id; (3) Build training dataset with point-in-time correct features (use feature values at decision time, not current values); (4) Daily model refresh using ~7-day labeled data (fast cycle, handles emerging patterns); (5) Monthly full retrain with 90-day confirmed labels (higher quality); (6) Shadow new model against production before cutover — compare fraud rate, FPR, score distribution; (7) Monitor for training-serving skew by comparing feature distributions at training time vs live traffic.

**Q: A fraudster is testing stolen cards with small $1 transactions before a large purchase. How do you detect this?**  
A: Card testing leaves velocity signatures: rapid succession of small transactions followed by a larger one. Features: txn_count_1h (spike of 10+ small transactions), amount_variance_1h (low then high), time_between_txns (very fast — bot-like). Sequence model captures this temporal pattern better than tabular GBM. Additionally: merchant category diversity in 1h (card testers try different merchants), micro-transaction flag (amount < $2 on a card that usually spends >$50). Rule: if txn_count_1h > 5 AND any_amount_under_$2 in last hour AND current_amount > $200 → escalate to manual review.

## Flashcards

**Why is accuracy a useless metric for fraud detection, even at 98%?** #flashcard
With a ~0.1% fraud rate, a model that always predicts "not fraud" gets 99.9% accuracy while catching zero fraud. PR-AUC and recall-at-fixed-FPR are the metrics that actually reflect fraud-catching performance on this class imbalance.

**Why use PR-AUC instead of ROC-AUC as the primary fraud metric?** #flashcard
ROC-AUC's false positive rate axis is diluted by the 99.9% of transactions that are true negatives, making it insensitive to changes that matter operationally; PR-AUC directly reflects the precision-recall trade-off at the low-FPR operating points fraud systems actually run at.

**Why does the fraud model use a two-model ensemble (GBT + sequence/neural) instead of one model?** #flashcard
GBT handles tabular, non-linear feature interactions well and is fast/interpretable via SHAP, but can't capture temporal patterns; a sequence model (LSTM over transaction history) is needed specifically for account-takeover patterns like new device → password change → new payee → large transfer, which only show up as an ordered sequence.

**Why can't fraud models be trained on chargeback labels alone?** #flashcard
Chargebacks take 30-90 days to arrive, so training only on confirmed fraud misses "silent fraud" that was never reported and starves the model of recent data. A label hierarchy (rule blocks → manual review → disputes → chargebacks) lets daily retraining use fast, lower-quality signals while monthly retraining uses slow, high-quality confirmed labels.

**Why must fraud model validation use a temporal split, never a random split?** #flashcard
Fraud patterns drift over time — new attack vectors emerge and old ones get patched — so a random split leaks future fraud patterns into training and overstates how well the model will generalize to genuinely new attacks it hasn't seen yet.

**Why combine a rules engine with an ML model instead of using ML alone?** #flashcard
Rules are fast, interpretable, and auditable (needed for regulatory compliance and obvious cases like stolen-card blocklists) but brittle since adversaries adapt to known rules; ML adapts to subtle patterns but is a black box and needs labeled data. Production systems run rules first for clear-cut blocks, then ML for the ambiguous middle.

**Why does `scale_pos_weight` (or focal loss) matter for training a fraud GBM, and what breaks without it?** #flashcard
At a ~1000:1 class imbalance, an unweighted model minimizes loss by mostly ignoring the minority (fraud) class, since predicting "not fraud" for everything is already near-optimal for unweighted loss — weighting the positive class (or down-weighting easy negatives via focal loss) forces the model to actually learn fraud-distinguishing signal.

**Why is a card-testing attack (many small transactions, then one large one) hard for a plain tabular GBM to catch but easier for a sequence model?** #flashcard
Any single small transaction looks unremarkable in isolation — the fraud signal is in the temporal pattern (rapid succession, low-then-high amount variance) across multiple transactions, which a per-transaction tabular model doesn't see but a sequence/velocity-feature model does.

**Why do retraining triggers for fraud models include both precision drops AND recall drops, not just one?** #flashcard
A precision drop signals a new fraud pattern is bypassing the model (missed positives look normal); a recall drop signals rising false negatives, e.g. from unreported fraud increasing the effective noise in "negative" labels — both indicate the model no longer matches current fraud behavior, but point to different causes.

**Why treat unconfirmed recent transactions (<90 days old) as censored rather than as negative labels?** #flashcard
A transaction with no chargeback yet isn't confirmed non-fraud — it may simply not have been reported yet within the observation window. Treating it as a hard negative teaches the model a wrong ground truth; survival-analysis framing (or excluding it until the window closes) avoids this bias.
