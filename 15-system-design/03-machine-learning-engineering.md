---
module: System Design
topic: System Design
subtopic: Machine Learning Engineering
status: unread
tags: [productionml, ml, system-design-machine-learning]
---
# Machine Learning Engineering

Based on Andriy Burkov's _Machine Learning Engineering_.

---

## What Machine Learning Engineering Is

**The problem**: research ML optimizes for accuracy on benchmarks. Production ML must also be reliable, maintainable, debuggable, and economically justified.

**The core insight**: an ML engineer converts business problems into ML solutions that work reliably in production — not chase state-of-the-art accuracy.

**The mechanics**: MLE lifecycle, nine stages:

```
1. Goal definition        4. Feature engineering    7. Model deployment
2. Data collection        5. Model training          8. Model serving
3. Data preparation       6. Model evaluation         9. Monitoring & maintenance
```

Each stage's output feeds the next; failures propagate forward. Most production failures start in stages 1-4 (problem framing, data), not model architecture.

**What breaks**: excellent accuracy on a metric that doesn't matter. 97% precision at 30% recall passes offline eval but misses 70% of opportunities — worthless commercially. Goal-definition failures propagate through every later stage.

---

## Project Prioritization

**The problem**: ML projects are expensive and uncertain. Teams spend months on projects that never needed ML, or that need accuracy that's economically unreachable.

**The core insight**: prioritize by impact/cost. Impact is high when ML replaces complex hand-coded logic, or fast-imperfect ML beats slow-perfect human judgment. Cost grows superlinearly with required accuracy.

**The mechanics**:

```
Required accuracy    Relative cost
80%                  1x
90%                  10x
95%                  30x
99%                  100x
99.9%                1000x
```

Going from 90% to 99% isn't twice as hard — it's ~10x as hard. Data collection, labeling, and edge cases each multiply cost per accuracy point. Use pilot projects on a data slice to validate feasibility before full commitment.

**What breaks**: teams anchor to accuracy targets set by stakeholders who don't see the cost curve. "We need 99%" costs 10x more than "we need 95%." Push back with the curve and let the business decide what it can afford.

---

## Problem Framing

**The problem**: business objectives ("reduce churn") aren't directly optimizable. ML needs a precise definition: input, output, training label, and metric.

**The core insight**: translate business objective → decision the model makes → prediction task.

```
Business objective          | ML task               | Label
-----------------------------|------------------------|----------------------
Reduce customer churn        | Binary classification  | churned in 30 days
Improve search relevance     | Ranking                | clicked / booked
Detect payment fraud         | Binary classification  | confirmed fraud
Predict inventory demand     | Regression             | units sold next week
Personalize recommendations  | Ranking                | engaged / purchased
```

Static (computed offline on a schedule) vs. dynamic (computed at request time) predictions — static is simpler and cheaper when freshness doesn't matter.

Define baselines before training: majority-class, rule-based, and simple linear model. If ML can't beat these, it shouldn't ship.

**What breaks**: optimizing a proxy metric that diverges from the business goal. A fraud model chasing AUC might block legitimate transactions and drive customers away. Track the real business KPI alongside the ML metric.

---

## Data Collection and Labeling

### Data Quality Dimensions

**The problem**: garbage data produces garbage models — but the garbage is often invisible until the model fails in production.

**The core insight**: data quality has six independent dimensions; each can fail alone.

```
Completeness       Key fields missing for subpopulations
Consistency         Same entity labeled differently over time
Validity            Values outside expected range/format
Uniqueness           Duplicates inflate class frequencies
Timeliness           Stale features (yesterday's price today)
Representativeness   Training data doesn't match serving distribution
```

Validate before training with an expectations library (e.g., Great Expectations): assert non-null rates, value ranges, valid categories, near-uniqueness of IDs — fail the pipeline if violated.

**What breaks**: missing data often correlates with the label (e.g., missing age skews by demographic). Dropping those rows biases the sample. Investigate why data is missing before handling it mechanically.

---

### Labeling

**The problem**: labels are inconsistent, expensive, slow, and can encode bias from a flawed protocol.

**The core insight**: measure inter-annotator agreement before trusting labels. If two experts disagree 30% of the time, that's your label noise floor — no model can beat it.

**The mechanics**: Cohen's kappa on a shared subset. Kappa > 0.8 = reliable; 0.6-0.8 = investigate disagreements; < 0.6 = fix the protocol before training.

Programmatic labeling (weak supervision, e.g. Snorkel) combines multiple noisy labeling functions (keyword rules, heuristics) into a probabilistic label model to scale labels cheaply.

**What breaks**: programmatic labels encode the combined noise of all labeling functions. Always validate against a small manually labeled gold set — acceptable noise depends on the task (medical diagnosis tolerates far less than product categorization).

---

### Data Leakage Prevention

**The problem**: features computed after the prediction moment sneak into training data. The model learns to use information unavailable at serving time — great offline, broken in production.

**The core insight**: every training feature must reflect only information available at prediction time.

```python
# WRONG: uses information that only exists after fraud is confirmed
df['is_disputed_merchant'] = df['merchant_id'].isin(confirmed_fraud_merchants)

# RIGHT: point-in-time snapshot
df['merchant_risk_score'] = df['merchant_id'].map(merchant_risk_scores_as_of_transaction_time)

# Time series: sort and split chronologically, never shuffle first
df = df.sort_values('timestamp')
train, val, test = df.iloc[:int(.6*len(df))], df.iloc[int(.6*len(df)):int(.8*len(df))], df.iloc[int(.8*len(df)):]

# Fit scalers/encoders on train only
scaler = StandardScaler().fit(train[feature_cols])
```

**What breaks**: target encoding (replacing a category with its average label) is a common subtle leak — fit it only on training folds, never the full dataset before splitting.

---

## Data Preparation

### Missing Value Treatment

**The problem**: most algorithms can't handle missing values; dropping rows biases the sample, naive imputation misleads the model.

**The core insight**: missingness is itself a signal — "missing" might mean "never purchased" or "pipeline error," each needing different treatment. Match strategy to mechanism: MCAR → mean/median; MAR → regression/KNN imputation; MNAR → add a missingness indicator column.

```python
def handle_missing(df, strategy='median'):
    for col in df.columns:
        if df[col].isnull().any():
            df[f'{col}_was_missing'] = df[col].isnull().astype(int)
    num_cols = df.select_dtypes(include='number').columns
    df[num_cols] = SimpleImputer(strategy=strategy).fit_transform(df[num_cols])
    cat_cols = df.select_dtypes(include='object').columns
    df[cat_cols] = df[cat_cols].fillna("UNKNOWN")
    return df
```

**What breaks**: KNN imputation is O(n^2) — infeasible above ~100K rows. Use median imputation with missingness indicators at scale.

---

### Outlier Treatment

**The problem**: extreme values distort regression coefficients, distance metrics, and gradient-based optimization.

**The core insight**: distinguish data errors (remove) from rare-but-valid extremes (retain, winsorize). Removing valid extremes biases the model; keeping errors adds noise.

```python
Q1, Q3 = df[col].quantile(.25), df[col].quantile(.75)
IQR = Q3 - Q1
df[col] = df[col].clip(lower=Q1 - 3*IQR, upper=Q3 + 3*IQR)  # winsorize, keep the row

df['log_amount'] = np.log1p(df['amount'])  # log transform for skewed features
```

**What breaks**: clip bounds computed at training time must be reused verbatim at serving time (not recomputed on serving data, which shifts). Store bounds as model metadata.

---

### Scaling and Encoding

**The problem**: gradient-based models are sensitive to feature scale — a 0-1M range feature dominates a 0-1 range one.

**The core insight**: scale on training data only, and apply identically at serving time.

- `StandardScaler` — Gaussian-ish features; `MinMaxScaler` — bound to [0,1] for sigmoid/tanh outputs.
- One-hot encoding for low-cardinality categoricals (<50 values); ordinal encoding for ordered categories.
- Cyclical encoding (`sin`/`cos` of `2π·value/period`) for periodic features like hour-of-day, so hour 23 and hour 0 stay close.

**What breaks**: scaler `mean_`/`scale_` must be saved with the model artifact (e.g., via a scikit-learn Pipeline), or serving can't reproduce training-time scaling.

---

## Feature Engineering

### Derived Features and Interactions

**The problem**: raw features often don't linearly predict the target; a linear model can't discover "loan/income ratio matters" on its own.

**The core insight**: domain knowledge encodes signal a model would otherwise need far more data to learn implicitly.

Common patterns: ratio features (`loan_to_income`), temporal features from timestamps (hour, weekend flag, recency), interaction terms, and user-level aggregations (avg/std/count) joined back onto each row, plus deviation-from-personal-baseline features.

**What breaks**: aggregations computed over full user history (including future transactions relative to the label event) leak. Compute aggregations with point-in-time joins only.

---

### Feature Selection

**The problem**: more features means more overfitting risk and compute cost; many are redundant, irrelevant, or just noise.

**The core insight**: fewer well-chosen features usually beat many poorly chosen ones.

- **Filter** (fast, model-agnostic): mutual information, drop highly correlated pairs (|r| > 0.9).
- **Wrapper** (accurate, expensive): recursive feature elimination.
- **Embedded**: L1/Lasso zeroes out irrelevant coefficients; tree-based feature importances.
- **SHAP** gives consistent, model-agnostic importance for final selection/explanation.

**What breaks**: importance from one model doesn't transfer to another — a feature important to a random forest may be irrelevant to a neural net that learns the same signal from raw inputs. Reevaluate when architecture changes.

---

## Model Training

### Loss Function Selection

**The problem**: the loss defines what the model optimizes. MSE for classification or cross-entropy for regression is valid math, wrong outcome.

**The core insight**: match the loss to the task structure and the cost of different error types.

```python
mse_loss = F.mse_loss(pred, target)              # regression, penalizes large errors
huber_loss = F.huber_loss(pred, target)           # regression, robust to outliers
bce_logit_loss = F.binary_cross_entropy_with_logits(logits, target)  # binary, numerically stable
weighted_bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=torch.tensor([99.0]))  # imbalanced
ce_loss = F.cross_entropy(logits, labels)         # multiclass
bpr_loss = -torch.log(torch.sigmoid(pos_scores - neg_scores)).mean()  # pairwise ranking
```

**What breaks**: loss and evaluation metric can diverge — training on differentiable cross-entropy but judged on non-differentiable F1 can yield good loss, mediocre F1. Use a surrogate loss that correlates with the target metric.

---

### Hyperparameter Tuning

**The problem**: hyperparameters aren't learned from data; wrong choices cause under/overfitting or non-convergence.

**The core insight**: tuning is nested optimization — outer loop searches hyperparameter space, inner loop trains and validates.

Bayesian optimization (e.g., Optuna) samples the next trial's hyperparameters using results from prior trials, far more sample-efficient than grid search. Learning-rate range tests (exponentially increase LR, pick the value just before loss diverges) are a fast pre-check.

**What breaks**: tuning against a small validation set overfits the hyperparameters to that set by chance. Use k-fold CV for tuning when data is limited, and reserve a separate holdout never touched during tuning for the final number.

---

### Preventing Overfitting

**The problem**: a high-capacity model memorizes training examples — near-zero training loss, high validation loss.

**The core insight**: total error = bias² + variance + noise. Overfitting is high variance — fix with more data, less capacity, or explicit regularization.

Diagnose with training-vs-validation loss curves (overfitting signature: train loss still falling, val loss rising). Early stopping: track best validation score, stop after `patience` epochs without improvement, restore the best checkpoint.

**What breaks**: early stopping only gives an unbiased estimate if the validation set wasn't also used for hyperparameter tuning. If both used the same val set, you need a separate final test set.

---

## Model Evaluation

### Reproducibility

**The problem**: identical code produces different models run-to-run because of random init, data order, and non-deterministic GPU ops.

**The core insight**: seed every randomness source — Python, NumPy, and framework RNGs — and set deterministic backend flags.

Track every experiment (params, metrics, code/data version) with a tool like MLflow, and version the dataset itself (e.g., DVC) so a run can be exactly reproduced.

**What breaks**: tracking code/params without data versioning is incomplete — two runs can silently use different data if the pipeline reran overnight. Track a data version hash with every experiment.

---

### Model Comparison and Error Analysis

**The problem**: a single aggregate metric hides regressions on important minority slices.

**The core insight**: a new model is only better if it doesn't regress on the slices that matter. Compare overall and per-slice metrics before shipping.

Error analysis: separate false positives/negatives, look for feature patterns that distinguish errors from correct predictions.

**What breaks**: error analysis surfaces patterns, not root causes. More errors on mobile users could mean a UI bug, a real behavioral difference, or sampling bias — distinguishing these needs human investigation.

---

## Deployment

### Deployment Paradigms

**The problem**: use cases differ wildly in latency/throughput needs. Fraud scoring needs <100ms; nightly risk scoring can take minutes per record.

**The core insight**: latency requirement picks the deployment mode — batch for offline scoring, online API for synchronous real-time decisions, streaming for continuous event processing.

Batch: load a model, broadcast it to distributed workers (e.g., Spark), score in bulk, write results. Online: wrap the model in a low-latency API (e.g., FastAPI) that extracts features and returns a decision per request.

**What breaks**: online serving latency at p99 is often dominated by a single slow feature lookup (e.g., a DB call that's 50ms average but 500ms p99), not model inference. Profile each feature source independently.

---

### Training-Serving Skew Prevention

**The problem**: features computed one way in training and another way at serving silently shift the input distribution the model sees.

**The core insight**: feature computation logic must be identical — same code, same library versions, same null handling — at training and serving time.

Use one versioned, tested feature-computation function shared by both paths, and package the model with its scaler/encoder/schema as a single artifact.

**What breaks**: don't inline feature logic in notebooks or rely on pickling closures. Put it in a versioned library both training and serving import.

---

## Monitoring and Maintenance

### Drift Detection

**The problem**: a model trained in January sees different data in July — demographics shift, behavior changes — and silently degrades.

**The core insight**: monitor input distributions continuously; don't wait for downstream metrics to decline.

- KS test per feature: flag when p-value < 0.05 vs. training distribution.
- PSI (Population Stability Index): <0.1 stable, 0.1-0.2 investigate, >0.2 retrain.

**What breaks**: these catch marginal feature drift, not concept drift (feature-label relationship changes while feature distributions look stable) — e.g., fraud tactics evolve. Also monitor prediction confidence and actual outcomes when available.

---

### Retraining Strategy

**The problem**: static models degrade; retraining too often wastes compute, too rarely lets the model go stale.

**The core insight**: retraining cadence should match how fast the domain shifts — fraud/markets need frequent retraining, stable domains rarely.

Trigger retraining on performance drop past a threshold or on feature PSI past a threshold. Choose training data window: rolling window (recent data, fast-changing domains) vs. full history (stable domains, rare events) vs. exponentially recency-weighted.

**What breaks**: a rolling window discards old rare events — if a fraud pattern from two years ago recurs, a 90-day-window model has never seen it. Balance recency against retaining rare-pattern knowledge.

---

### Model Versioning and Rollback

**The problem**: a new model version degrades in production; without versioned artifacts, recovery means retraining from scratch.

**The core insight**: every production model needs a versioned registry entry, logged metrics, and a tested rollback path.

Promote a candidate to production only if it beats the current production model by a set threshold (e.g., AUC improvement); archive the outgoing version instead of deleting it, so rollback is a stage transition, not a rebuild.

**What breaks**: rollback without root-cause analysis just defers the problem — if the new model failed due to a bad upstream data pipeline, the old model is exposed to the same bad data. Rollback buys time; it isn't a fix.

---

## Ethics and Privacy

### Bias and Fairness

**The problem**: models trained on historical data inherit historical bias. Removing protected attributes doesn't fix it — correlated proxies (zip code, name, school) carry the same signal.

**The core insight**: fairness must be measured and enforced explicitly. Different fairness definitions (demographic parity, equal opportunity, equalized odds) are mathematically incompatible when base rates differ across groups — pick one deliberately.

Compute per-group positive rate, TPR, FPR; check the disparate impact ratio (group B rate / group A rate) against the 80% rule.

**What breaks**: Chouldechova's impossibility result shows demographic parity, equal opportunity, and calibration can't all hold at once when base rates differ. Document which criterion you chose and why.

---

### Privacy Preservation

**The problem**: models can memorize training data — membership inference or extraction attacks can recover it.

**The core insight**: privacy techniques (differential privacy, federated learning) trade accuracy for privacy; calibrate the tradeoff deliberately rather than defaulting.

Differential privacy adds calibrated noise to gradients during training (e.g., via Opacus), controlled by a privacy budget (epsilon) — smaller epsilon means stronger privacy but more accuracy loss. Federated learning trains locally on-device and shares only weight updates, never raw data.

**What breaks**: strong privacy (very small epsilon) can add so much gradient noise the model fails to converge; very weak privacy makes the guarantee nearly meaningless. Calibrate epsilon to the actual sensitivity of the data, not a default.

## Flashcards

**Static?** #flashcard
compute offline on a schedule (e.g., nightly churn scores)

**Dynamic?** #flashcard
compute at request time (e.g., real-time fraud score)
