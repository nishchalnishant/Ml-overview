# Additional ML interview topics

High-signal questions that often come up in **machine learning** interviews—complementing the other files in this folder.

---

# Q1: What is data leakage, and how do you detect and prevent it?

## 1. 🔹 Direct Answer
**Data leakage** is when **information from the target or future** (or from the test distribution) **inappropriately** enters **training** features, making **offline metrics** unrealistically good and **production** performance poor. **Prevent** by **time-aware** splits, **pipeline** isolation (fit transformers on train only), and **auditing** features for **post-outcome** variables.

## 2. 🔹 Intuition
The model “**cheats**” by seeing the answer (or something correlated with it) in disguise.

## 3. 🔹 Deep Dive
- **Train-test contamination**: scaling on full data, **duplicate** rows across splits.
- **Temporal leakage**: aggregates that include **future** events (e.g., “lifetime spend” computed with tomorrow’s data).
- **Target leakage**: engineered feature **only** known **after** label (e.g., “doctor’s final diagnosis” when predicting **early** triage).

## 4. 🔹 Practical Perspective
**Ablation**: drop suspicious feature—if AUC **collapses**, investigate. **Holdout** by **entity** (user, patient) not only row.

## 5. 🔹 Code Snippet
```python
from sklearn.pipeline import Pipeline
pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression())])
pipe.fit(X_train, y_train)  # scaler sees only train
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Kaggle leaderboard vs real world? **A:** Often leakage or duplicate near-duplicates across splits.
2. **Q:** Cross-validation still leaks? **A:** Yes, if preprocessing uses **global** stats outside each fold.

## 7. 🔹 Common Mistakes
Random split on **time-series** or **grouped** data (same patient in train and test).

## 8. 🔹 Comparison / Connections
Overfitting to validation set (multiple peeking).

## 9. 🔹 One-line Revision
Leakage smuggles label or future information into features—use causal feature timing and proper CV pipelines.

## 10. 🔹 Difficulty Tag
🟣 Hard

---

# Q2: Explain SMOTE. When does it help, and what are the risks?

## 1. 🔹 Direct Answer
**SMOTE** (**Synthetic Minority Over-sampling**) creates **synthetic** minority examples by **interpolating** between **k** nearest minority neighbors in feature space—**balances** class distribution without pure duplication. **Risks**: **blurs** boundaries in **noisy** data, can **amplify** label errors, **invalid** if **features** are **categorical** without care; **never** apply to **test** set.

## 2. 🔹 Intuition
“**Between** real minority points” as plausible new positives—only if **linear** neighborhoods make sense.

## 3. 🔹 Deep Dive
Variants: **Borderline-SMOTE**, **ADASYN** focus on harder regions.

## 4. 🔹 Practical Perspective
Compare against **class weights** and **threshold** tuning first—often **simpler** and **less** artifact-prone.

## 5. 🔹 Code Snippet
```python
from imblearn.over_sampling import SMOTE
X_res, y_res = SMOTE(random_state=42).fit_resample(X_train, y_train)
```

## 6. 🔹 Interview Follow-ups
1. **Q:** High-dimensional sparse text? **A:** SMOTE often **poor**—distance meaningless.

## 7. 🔹 Common Mistakes
Applying SMOTE **before** CV so **synthetic** points leak across folds.

## 8. 🔹 Comparison / Connections
Undersampling majority, focal loss.

## 9. 🔹 One-line Revision
SMOTE synthesizes minority points along local neighborhoods—use inside training folds only and validate against simpler baselines.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q3: Batch Normalization vs Layer Normalization—when do you use each?

## 1. 🔹 Direct Answer
**BatchNorm**: normalizes **across batch** per channel (CNN) or feature (MLP)—**strong** regularization from batch noise; needs **reasonable batch size**; **different** train (batch stats) vs eval (running averages). **LayerNorm**: normalizes **across features** **per sample**—**independent** of batch size; **default** in **Transformers** (sequence length axis or hidden dim depending on convention).

## 2. 🔹 Intuition
BatchNorm: “how does this **channel** compare to **others** in this batch?” LayerNorm: “normalize this **token’s** vector **before** the next sublayer.”

## 3. 🔹 Deep Dive
**GroupNorm / InstanceNorm** interpolate for small-batch vision.

## 4. 🔹 Practical Perspective
**SyncBatchNorm** for multi-GPU consistency; **RMSNorm** (LLaMA) simplifies LayerNorm.

## 5. 🔹 Code Snippet
```python
nn.BatchNorm2d(c); nn.LayerNorm(d_model)
```

## 6. 🔹 Interview Follow-ups
1. **Q:** BN in RNN? **A:** Awkward—LayerNorm per timestep more common.

## 7. 🔹 Common Mistakes
Using BatchNorm with **batch size 1** in training.

## 8. 🔹 Comparison / Connections
Weight standardization, pre-norm Transformer blocks.

## 9. 🔹 One-line Revision
BatchNorm uses batch statistics (CNNs, large batches); LayerNorm is per-example across features (Transformers, variable batch).

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q4: Compare XGBoost, LightGBM, and CatBoost in one interview-ready answer.

## 1. 🔹 Direct Answer
All are **gradient-boosted trees** with regularization. **LightGBM**: **leaf-wise** growth, **histogram** splits—**very fast** on large data, can **overfit** without tuning. **XGBoost**: **level-wise**, **histogram** option, **mature** ecosystem—**strong default**. **CatBoost**: **ordered boosting** and **native** **categorical** handling—**less** tuning for **many** cats, **slower** sometimes.

## 2. 🔹 Intuition
Pick **LightGBM** for **speed** and huge tabular; **CatBoost** when **categories** dominate; **XGBoost** for **balanced** competitions and **documentation**.

## 3. 🔹 Deep Dive
All support **monotonic constraints**, **missing** value handling (implementation differs).

## 4. 🔹 Practical Perspective
**Benchmark** on **your** data with **same CV**—no universal winner.

## 5. 🔹 Code Snippet
```python
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Why leaf-wise risk? **A:** Can grow **deep** asymmetric trees—needs **max_depth** / **num_leaves** control.

## 7. 🔹 Common Mistakes
Claiming one is “always” best without ablation.

## 8. 🔹 Comparison / Connections
Random Forest (bagging), NGBoost.

## 9. 🔹 One-line Revision
LightGBM fast leaf-wise; XGBoost reliable all-rounder; CatBoost strong on categoricals—benchmark empirically.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q5: DBSCAN vs k-means—when do you choose which?

## 1. 🔹 Direct Answer
**k-means** assumes **roughly spherical** clusters, **fixed k**, minimizes within-cluster variance. **DBSCAN** finds **arbitrary-shaped** clusters via **density**, discovers **noise** points, **does not** require **k**—needs **distance** threshold **ε** and **minPts**. Use **DBSCAN** for **irregular** geometry and **outliers**; **k-means** for **speed** and **vector quantization** when **spherical** assumption OK.

## 2. 🔹 Intuition
DBSCAN: “**crowded** regions are clusters, **loners** are noise.” k-means: “**k** balloon centers.”

## 3. 🔹 Deep Dive
DBSCAN **struggles** with **varying densities** (HDBSCAN helps).

## 4. 🔹 Practical Perspective
**Scale** features; **curse of dimensionality** hurts both—**PCA** first sometimes.

## 5. 🔹 Code Snippet
```python
from sklearn.cluster import DBSCAN
DBSCAN(eps=0.5, min_samples=5)
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Optics? **A:** Hierarchical density extension—ordering of points.

## 7. 🔹 Common Mistakes
Using **Euclidean** on **categorical** data without embedding.

## 8. 🔹 Comparison / Connections
Spectral clustering, hierarchical agglomerative.

## 9. 🔹 One-line Revision
k-means for spherical partitions and fixed k; DBSCAN for density-based clusters and noise—neither solves all shapes.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q6: Collaborative filtering vs content-based recommendation.

## 1. 🔹 Direct Answer
**Collaborative filtering** uses **behavior** (ratings, clicks) across users—**“users like you liked X.”** **Content-based** uses **item features** (genre, text, image)—**“items like this one.”** **Hybrid** systems dominate production (Netflix-style).

## 2. 🔹 Intuition
Collaborative: **wisdom of crowds**; content: **similarity of objects**.

## 3. 🔹 Deep Dive
**Matrix factorization** / **ALS** for collaborative; **embedding** items from **metadata** for content. **Cold start**: new user → content; new item → content or **popularity**.

## 4. 🔹 Practical Perspective
**Implicit feedback** (clicks) needs **negative sampling** or **pairwise** losses.

## 5. 🔹 Code Snippet
```text
minimize ||R - UVᵀ|| + λ(||U||² + ||V||²)  # matrix factorization
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Popularity bias? **A:** **IPS** / **inverse propensity** for unbiased learning from logs.

## 7. 🔹 Common Mistakes
Ignoring **position bias** in click logs.

## 8. 🔹 Comparison / Connections
Two-tower models, graph neural recommenders.

## 9. 🔹 One-line Revision
Collaborative uses user-item interactions; content-based uses item attributes—combine for cold start and accuracy.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q7: What is model calibration, and how do you fix miscalibration?

## 1. 🔹 Direct Answer
**Calibration**: predicted **probabilities** match **empirical** frequencies (e.g., among predictions of **0.7**, ~70% should be positive). **Measure** with **reliability diagrams**, **ECE** (expected calibration error). **Fix**: **temperature scaling** (single **T** on logits), **Platt scaling**, **isotonic regression** (flexible, needs data).

## 2. 🔹 Intuition
**Accuracy** can hide **overconfident** scores—bad for **thresholds** and **downstream** decisions.

## 3. 🔹 Deep Dive
**Modern NNs** often **overconfident**—**post-hoc** calibration on **validation** set.

## 4. 🔹 Practical Perspective
**Class imbalance** skews raw probabilities—**rebalance** or **calibrate** per class slice.

## 5. 🔹 Code Snippet
```python
from sklearn.calibration import CalibratedClassifierCV
cal = CalibratedClassifierCV(model, method="isotonic", cv=3)
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Calibrate after threshold tuning? **A:** Calibrate **first** on held-out set, then pick **cost-based** threshold.

## 7. 🔹 Common Mistakes
Calibrating on **test** set—**optimistic**.

## 8. 🔹 Comparison / Connections
Brier score, proper scoring rules.

## 9. 🔹 One-line Revision
Calibration aligns predicted probabilities with reality—use temperature/isotonic on a validation set, measure ECE.

## 10. 🔹 Difficulty Tag
🟣 Hard

---

# Q8: How should you validate time-series models differently from i.i.d. data?

## 1. 🔹 Direct Answer
**No random shuffle**—use **chronological** **train/val/test** or **rolling / walk-forward** CV: train on **past**, validate on **next** **window**. **Features** must use **only past** information at each time (**no future** aggregates). **Metrics** on **multiple** horizons.

## 2. 🔹 Intuition
Random split lets the model **peek** at the **future** of the series—fake performance.

## 3. 🔹 Deep Dive
**Purged** gaps if labels overlap in time; **GroupKFold** by entity if **multiple** series.

## 4. 🔹 Practical Perspective
**Seasonality** and **regime change**—monitor **post-deployment** drift.

## 5. 🔹 Code Snippet
```python
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
for train_idx, val_idx in tscv.split(X):
    ...
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Global model vs per-series? **A:** Hierarchical forecasting, **meta-learning**.

## 7. 🔹 Common Mistakes
**StandardScaler** fit on **entire** series including **future**.

## 8. 🔹 Comparison / Connections
Causal inference, leakage.

## 9. 🔹 One-line Revision
Time-series validation respects order—rolling windows and purging; never shuffle time if causality matters.

## 10. 🔹 Difficulty Tag
🟣 Hard

---

# Q9: How are KL divergence and cross-entropy related in classification?

## 1. 🔹 Direct Answer
For true distribution **p** (often one-hot) and predicted **q**, **cross-entropy H(p,q) = −Σ p_i log q_i**. **KL(p||q) = Σ p_i log(p_i/q_i) = H(p,q) − H(p)**. Since **H(p)** is **constant** for fixed labels, **minimizing CE** is **equivalent** to minimizing **KL** to the **empirical** label distribution—**MLE** for **multinomial** with softmax.

## 2. 🔹 Intuition
CE measures **coding length** under **q** when truth is **p**; KL is **extra** bits vs using **true** **p**.

## 3. 🔹 Deep Dive
**Label smoothing** moves **p** from one-hot—**prevents** overconfident **q**.

## 4. 🔹 Practical Perspective
**torch.nn.CrossEntropyLoss** = softmax + CE in one stable kernel.

## 5. 🔹 Code Snippet
```python
# minimizing CE on one-hot y is MLE for categorical model
loss = -torch.log(q[y_true]).mean()
```

## 6. 🔹 Interview Follow-ups
1. **Q:** KL vs CE for training? **A:** Same gradient w.r.t. **q** up to **H(p)** constant.

## 7. 🔹 Common Mistakes
Thinking **minimizing CE** is unrelated to **information theory**.

## 8. 🔹 Comparison / Connections
Maximum likelihood, perplexity.

## 9. 🔹 One-line Revision
Cross-entropy loss equals KL divergence to targets up to an additive constant—standard classification objective.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q10: Permutation importance vs SHAP—what’s the difference?

## 1. 🔹 Direct Answer
**Permutation importance**: **shuffle** one feature (or column), measure **drop** in **validation score**—**fast**, **model-agnostic**, **global** importance. **SHAP** (**Shapley**): **fair** attribution of **prediction** difference to features based on **game-theoretic** averaging over **coalitions**—**local** explanations per **row**, **expensive** for large models.

## 2. 🔹 Intuition
Permutation: “**break** this feature and see if the model cares.” SHAP: “**fair share** of blame/credit among features for **this** prediction.”

## 3. 🔹 Deep Dive
**TreeSHAP** exact for trees; **KernelSHAP** model-agnostic sampling.

## 4. 🔹 Practical Perspective
Use **permutation** for **ranking** features in **tabular**; **SHAP** for **customer-facing** **explanations** (with latency budget).

## 5. 🔹 Code Snippet
```python
from sklearn.inspection import permutation_importance
r = permutation_importance(model, X_val, y_val, n_repeats=10)
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Correlated features? **A:** Permutation can **split** importance arbitrarily—**group** permutations or **regularize**.

## 7. 🔹 Common Mistakes
**Permuting** **before** train-test split leakage if done wrong—must be **on val** only.

## 8. 🔹 Comparison / Connections
LIME, integrated gradients.

## 9. 🔹 One-line Revision
Permutation importance is cheap global knock-out tests; SHAP gives principled local attributions—costlier.

## 10. 🔹 Difficulty Tag
🟣 Hard

---

# Q11: What is label smoothing, and why is it used in deep classification?

## 1. 🔹 Direct Answer
Replace **one-hot** target **y** with **y_smooth = (1−ε) y + ε/K** for **K** classes—**softens** targets. **Reduces overconfidence**, acts as **regularizer**, can **improve** **generalization** and **calibration**; common in **Transformers** and **distillation**.

## 2. 🔹 Intuition
Tell the model “**maybe** not **100%** sure of this class”—**discourages** **extreme** logits.

## 3. 🔹 Deep Dive
**ε** typically **0.05–0.1**; **adjust** loss accordingly (still proper **mixture** target).

## 4. 🔹 Practical Perspective
**Too much** smoothing **hurts** when labels are **reliable**.

## 5. 🔹 Code Snippet
```python
# PyTorch CrossEntropyLoss label_smoothing=0.1
nn.CrossEntropyLoss(label_smoothing=0.1)
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Relation to KL to uniform? **A:** Smoothing pulls toward **uniform** prior on wrong mass.

## 7. 🔹 Common Mistakes
Using large **ε** on **tiny** datasets with **clean** labels.

## 8. 🔹 Comparison / Connections
Confidence penalty, mixup.

## 9. 🔹 One-line Revision
Label smoothing softens one-hot targets to reduce overfitting and overconfidence—standard in modern classifiers.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q12: What is the difference between precision@k, recall@k, and MAP?

## 1. 🔹 Direct Answer
**Precision@k**: fraction of **top-k** predicted items that are **relevant**. **Recall@k**: fraction of **all relevant** items captured in **top-k**. **MAP** (mean average precision): average of **precision at each relevant rank**, averaged over queries—**ranking** quality for **multiple** relevant docs.

## 2. 🔹 Intuition
P@k: “**quality** of the short list.” R@k: “**coverage** with k slots.” MAP: rewards **putting** relevant items **early**.

## 3. 🔹 Deep Dive
**NDCG** adds **graded** relevance and **position** discount.

## 4. 🔹 Practical Perspective
**k** must match **UI** (e.g., **10** results above fold).

## 5. 🔹 Code Snippet
```python
def precision_at_k(rec, rel, k):
    return len(set(rec[:k]) & rel) / k
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Optimize MAP directly? **A:** **Listwise** losses approximated; **LambdaMART** historically.

## 7. 🔹 Common Mistakes
Using **accuracy** for **ranking** tasks.

## 8. 🔹 Comparison / Connections
Recsys metrics, learning-to-rank.

## 9. 🔹 One-line Revision
P@k and R@k are top-k precision/recall; MAP averages precision across relevant positions for ranking eval.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q13: What is positive-unlabeled (PU) learning?

## 1. 🔹 Direct Answer
**PU learning**: only **positive** labels are trusted; **unlabeled** may contain **hidden positives** (e.g., fraud, disease). **Approaches**: **reweight** unlabeled as **mixture**, **biased SVM**, **nnPU** risk estimator, **two-step** (spy positives). **Do not** treat unlabeled as **definite negatives** without assumption.

## 2. 🔹 Intuition
**Absence of label ≠ negative** example.

## 3. 🔹 Deep Dive
**SCAR** assumption (selected completely at random) simplifies **estimation**—often **violated**.

## 4. 🔹 Practical Perspective
**Medical** screening, **fraud** where most **unlabeled** are **unknown** fraud.

## 5. 🔹 Code Snippet
```text
L = π E_{P+}[ℓ] + E_U[max(0, ℓ − g)]  # sketch: nnPU style ideas
```

## 6. 🔹 Interview Follow-ups
1. **Q:** vs semi-supervised? **A:** PU is **asymmetric** label noise structure.

## 7. 🔹 Common Mistakes
Training **binary** classifier with **unlabeled as class 0**.

## 8. 🔹 Comparison / Connections
Learning from noisy labels, anomaly detection.

## 9. 🔹 One-line Revision
PU learning trains when only positives are labeled and unlabeled is mixed—needs assumptions, not naive negative labeling.

## 10. 🔹 Difficulty Tag
🟣 Hard

---
