# PART 4: DATA DECISION FRAMEWORK

## Goal
To teach candidates how to diagnose and treat data problems systematically before touching any model code. The data decision is where most production failures originate.

## Mental Model
**Data is the fuel. The model is the engine. Even a perfect engine fails on bad fuel.**
Never reach for a new model when you haven't fixed your data. A 5% improvement in data quality often beats a 20% improvement in model complexity.

---

## 4.1 Missing Values

### Decision Tree
```text
Are values missing?
├── MCAR (Missing Completely At Random) → Simple imputation is safe.
│   └── Strategy: Mean/Median (numeric), Mode (categorical), Flag column.
├── MAR (Missing At Random) → Impute using correlated features.
│   └── Strategy: KNN Imputation, Iterative Imputer (MICE).
└── MNAR (Missing Not At Random) → Missingness IS signal!
    └── Strategy: Add binary "is_missing" flag feature. Do NOT impute blindly.
```

### When to Use What
| Technique | Use When | Risk |
| :--- | :--- | :--- |
| **Drop Rows** | < 1% data affected | Data loss |
| **Mean/Median** | Numeric, MCAR | Distorts variance, hides patterns |
| **Mode** | Categorical, low cardinality | Overrepresents one class |
| **KNN Impute** | MAR, moderate dataset | Computationally expensive |
| **MICE** | Complex missing patterns | Slow, risk of overfitting |
| **"Missing" category** | MNAR categorical | Simple, preserves signal |

### Common Mistake
Imputing missing values *before* train/test split using the entire dataset. **Fit the imputer on training data only, then transform validation/test data.** This is a classic data leakage source.

---

## 4.2 Outliers

### Decision Tree
```text
Are outliers present?
├── Are they ERRORS (data entry mistake)? → Remove or correct.
├── Are they SIGNAL (rare-but-real events)? → Preserve for fraud/anomaly detection.
└── Are they NOISE (sensor glitches)? → Clip to percentile (e.g., 1st–99th).
```

### Techniques
| Technique | Use When |
| :--- | :--- |
| **Z-score (> 3 std)** | Normally distributed data |
| **IQR Method** | Skewed distributions |
| **Log Transform** | Right-skewed features (income, price) |
| **Robust Scalers** | Use median/IQR instead of mean/std |
| **Winsorization** | Cap, don't remove |

---

## 4.3 Feature Engineering

### Decision Framework
1. **Domain knowledge first:** Always consult domain experts before engineering features.
2. **Time-based features:** Hour, day-of-week, days since last event, recency.
3. **Interaction features:** Product of two correlated features.
4. **Aggregation features:** User's average spend in the last 30 days.
5. **Embedding features:** User/item embeddings from a trained model, used as features for a downstream model.

### Common Mistake
Engineering features using future data (look-ahead bias). Example: Using a player's final season stats to predict if they'll churn *in* that season.

---

## 4.4 Categorical Encoding

### Decision Tree
```text
What is the cardinality?
├── LOW (<= 10 categories, no order) → One-Hot Encoding (OHE)
├── HIGH (> 10 categories) →
│   ├── Is there a meaningful order? → Ordinal Encoding
│   ├── Tree-based model? → Target Encoding (+ cross-val to prevent leak)
│   └── Neural Network? → Embedding Layer
└── EXTREMELY HIGH (user IDs, zip codes) → Hashing Trick / Embeddings
```

| Technique | Best For | Risk |
| :--- | :--- | :--- |
| **OHE** | Low-cardinality, linear models | Dimensionality explosion |
| **Ordinal** | Ordered categories (small, medium, large) | Implies incorrect numeric distance |
| **Target Encoding** | High-cardinality with tree models | Data leakage if not done within CV fold |
| **Embeddings** | High-cardinality with neural networks | Needs training, cold-start problem |

---

## 4.5 Normalization vs. Scaling

### Decision Tree
```text
What model type?
├── TREE-BASED (XGBoost, RF) → No scaling needed (splits are rank-based).
├── LINEAR / SVM / KNN / Neural Nets → Scaling REQUIRED.
│   ├── Normal distribution? → Standard Scaler (zero mean, unit variance).
│   ├── Skewed / Outliers? → Robust Scaler (median-based).
│   └── Bounded range needed (0-1)? → MinMax Scaler.
└── Distance-based (KNN, PCA) → Scaling REQUIRED.
```

### Common Mistake
Fitting the scaler on the full dataset (including test), leaking test statistics into training.

---

## 4.6 Class Imbalance

### Decision Tree
```text
What is the imbalance ratio?
├── MILD (1:4) → Adjust class_weight parameter in the model.
├── MODERATE (1:10 to 1:100) →
│   ├── Oversample minority → SMOTE
│   ├── Undersample majority → RandomUnderSampler
│   └── Use both → SMOTEENN / SMOTETomek
└── SEVERE (1:1000+) → Treat as anomaly detection.
    └── One-class SVM, Isolation Forest, Autoencoders.
```

| Technique | Use When | Risk |
| :--- | :--- | :--- |
| **class_weight** | Any model that supports it | Simple, first thing to try |
| **Random Oversampling** | Small dataset | Exact duplicates, may overfit |
| **SMOTE** | Tabular data, moderate imbalance | Generates unrealistic synthetic points |
| **Undersampling** | Huge majority class, speed matters | Loses potentially useful data |
| **Focal Loss** | Deep learning, class imbalance | Needs tuning of gamma parameter |

---

## 4.7 Text Preprocessing

### Decision Tree
```text
Are you using a pre-trained Transformer (BERT, etc.)?
├── YES → Let the tokenizer handle it. Minimal preprocessing.
│   └── Only remove: HTML tags, invisible chars.
└── NO (TF-IDF, FastText) →
    ├── Lowercase
    ├── Remove punctuation
    ├── Remove stopwords
    ├── Stemming or Lemmatization
    └── Handle negations (e.g., "not good" should not lemmatize to ["not", "good"])
```

### Common Mistake
Applying heavy preprocessing (stopword removal, stemming) before feeding into BERT. Transformers need the original text context, including punctuation.

---

## 4.8 Time Series Preprocessing

### Checklist
- [ ] **Stationarity check:** Apply ADF test. Apply differencing if non-stationary.
- [ ] **Temporal split:** NEVER use K-Fold CV. Use expanding window or sliding window.
- [ ] **Lag features:** Previous N timesteps as features.
- [ ] **Resampling:** Upsample (interpolate) or downsample (aggregate) as needed.
- [ ] **Anomaly treatment:** Segment-based, not point-based.

---

## 4.9 Streaming Data

### Decision Framework
- **Micro-batch (Spark Structured Streaming):** Process new data every N seconds. Lower complexity than true streaming.
- **True Streaming (Flink, Kafka Streams):** Sub-second latency, stateful processing. Use for fraud detection, live game events.
- **Key consideration:** How do you handle **late arriving data**? Define watermarks to decide when to close a window.

---

## 4.10 Data Leakage

### Types
1. **Target Leakage:** Feature contains information that is only available *after* the target is known. (e.g., using "refund_processed=True" to predict "will user request refund?")
2. **Temporal Leakage:** Using future data to predict the past. (e.g., fitting a scaler on the full dataset before train/test split.)
3. **Group Leakage:** Same user/patient in both train and test sets.

### Detection
```text
Suspiciously high CV accuracy → Suspect data leakage.
→ Check if any feature has near-perfect correlation with the target.
→ Check the temporal order of features vs. labels.
→ Check if any feature was computed using the full dataset.
```

---

## 4.11 Concept Drift vs. Data Drift

| Type | Definition | Detection | Remedy |
| :--- | :--- | :--- | :--- |
| **Data/Feature Drift** | Input distribution P(X) changes. | KS test, PSI (Population Stability Index). | Retrain with recent data. |
| **Concept Drift** | The relationship P(Y\|X) changes. | Business metric degradation (e.g., CTR drops). | Retrain, add new features that capture the new pattern. |
| **Label Drift** | Output distribution P(Y) changes. | Monitor prediction distribution. | Investigate root cause; labels may have changed. |

---

## Production Considerations

- **Feature Stores (Feast, Tecton):** Centralize feature computation to ensure training and serving use identical logic, eliminating training-serving skew.
- **Data Versioning (DVC):** Track changes to datasets, similar to how Git tracks code, to enable reproducibility.
- **Label Freshness:** Design pipelines to handle delayed labels (e.g., fraud chargebacks that arrive 60 days later) using deferred join patterns.

## Interview Follow-up Questions & Best Answers

**Q: "You see that your model's performance dropped overnight. Where do you look first?"**
*Best Answer:* "I follow a top-down triage. First, system metrics: Did latency spike? Is the model throwing errors? If the system is healthy, I check data metrics: Is there a feature distribution shift? I use PSI or KS test on today's input distributions vs. yesterday's baseline. If features look fine, I check business metrics to confirm performance truly dropped versus a measurement issue. Once I confirm the source, I either roll back (if a bad deployment), trigger a retrain (if drift), or fix the upstream data pipeline (if data corruption)."
