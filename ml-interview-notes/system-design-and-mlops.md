# System Design and MLOps

---

# Q1–Q13: End-to-end ML system design patterns (recommendation, search, feed, safety, multimodal, ads, delivery, image search, friends, e‑commerce rec)

## 1. 🔹 Direct Answer (framework)
For any large system: **clarify** requirements (**scale, latency, freshness, fairness**), **data** sources (**online logs, user/item features**), **offline** **training** pipeline, **serving** (**candidate generation → ranking → re-ranking**), **evaluation** (**offline metrics + online A/B**), **monitoring** (**drift, quality, business KPIs**).

## 2. 🔹 Intuition
**Two-stage** common: **cheap recall** (ANN, inverted index) then **expensive ranker** (GBDT/DNN). **Feedback loops** and **position bias** need handling.

## 3. 🔹 Deep Dive
- **YouTube-style rec**: **candidate generation** (collaborative, content), **deep ranker** on **hundreds** of features, **exploration**.
- **Search**: **query understanding**, **retrieval** (BM25+semantic), **LTR**.
- **Feed**: **freshness**, **diversity**, **dwell time** optimization—not just CTR.
- **Harmful content**: **human labels**, **active learning**, **multi-modal** models, **policy** layers, **appeals**.
- **Similar listings**: **embedding** items, **ANN** index.
- **Replacement rec**: **basket context**, **co-purchase** graphs.
- **Event rec**: **geo-temporal** features, **cold start** for new events.
- **Multimodal search**: **unified embedding** space (CLIP-like), **hybrid** retrieval.
- **Ad click**: **calibration** of pCTR, **fraud** filtering, **auction** integration.
- **Delivery time**: **regression** + **uncertainty**, **ops** constraints.
- **Image search**: **CNN/ViT embeddings**, **dedup**, **NSFW** filters.
- **Friends rec**: **graph** features, **privacy** constraints, **mutual** signals.
- **E-commerce rec**: **inventory**, **price**, **margin** aware objectives.

## 4. 🔹 Practical Perspective
Always discuss **cold start**, **bias** (popularity), **latency SLO**, **failure** modes (default ranking), **privacy**.

## 5. 🔹 Code Snippet
```text
offline: feature store -> train -> eval ; online: log -> features -> model -> rank
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Explore/exploit? **A:** Multi-armed bandits, ε-greedy on top of ranker.

## 7. 🔹 Common Mistakes
Only talking **model** without **data pipeline** and **serving**.

## 8. 🔹 Comparison / Connections
RAG systems, ads stack, search infra.

## 9. 🔹 One-line Revision
Large-scale ML systems pair retrieval + ranking + business constraints with strong eval and monitoring.

## 10. 🔹 Difficulty Tag
🟣 Hard

---

# Q14: How would you build a system to detect fraudulent transactions?

## 1. 🔹 Direct Answer
**Imbalanced** classification with **latency** SLO—**features**: user history, device, velocity, merchant category, **graph** signals. **Real-time** scoring + **rules engine** for **hard** blocks. **Human** review queue. **Metrics**: **precision@k**, **catch rate**, **$ loss** prevented.

## 2. 🔹 Intuition
**Arms race**—models + **expert rules** + **rapid** iteration.

## 3. 🔹 Deep Dive
**Sequence** models for **behavior**; **anomaly** detection baseline.

## 4. 🔹 Practical Perspective
**False positives** anger users—**cost-sensitive** thresholds.

## 5. 🔹 Code Snippet
```python
# sketch: score + rule overrides
if rules.hard_block(tx): return BLOCK
return "ALLOW" if model.predict_proba(feats) < tau else REVIEW
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Adversarial merchants? **A:** Concept drift—continuous labeling.

## 7. 🔹 Common Mistakes
Optimizing **accuracy** on 99.9% legit txs.

## 8. 🔹 Comparison / Connections
Anomaly detection, graph ML.

## 9. 🔹 One-line Revision
Fraud systems blend rules, real-time ML features, and human review with imbalance-aware metrics.

## 10. 🔹 Difficulty Tag
🟣 Hard

---

# Q15: Multimodal Fusion—Early vs Late Fusion.

## 1. 🔹 Direct Answer
**Early fusion**: **concatenate** raw features or **joint embedding** **before** deeper layers—**interactive** patterns, **harder** training. **Late fusion**: **separate** encoders then **combine scores/logits**—**modular**, **easier** to update one modality, **misses** low-level interactions.

## 2. 🔹 Intuition
Early = cook **ingredients together**; late = **separate dishes** then **combine** on plate.

## 3. 🔹 Deep Dive
**Intermediate** fusion (cross-attention) common in **Transformers**.

## 4. 🔹 Practical Perspective
**Missing modality** handling easier in late fusion with **defaults**.

## 5. 🔹 Code Snippet
```text
early: h = f([e_v, e_t]) ; late: y = σ(w_v s_v + w_t s_t)
```

## 6. 🔹 Interview Follow-ups
1. **Q:** CLIP? **A:** Contrastive joint embedding—early-ish fusion via shared space.

## 7. 🔹 Common Mistakes
Ignoring **alignment** of modalities (time sync).

## 8. 🔹 Comparison / Connections
Cross-attention, FiLM conditioning.

## 9. 🔹 One-line Revision
Early fusion learns joint representations; late fusion ensembles modality experts—trade interaction vs modularity.

## 10. 🔹 Difficulty Tag
🟣 Hard

---

# Q16–Q22: Applied ML scenarios (time series, spam, image classification, sentiment, churn, ranking, anomaly traffic, algorithm choice)

## Q16 Time series forecasting
**Decomposition** (trend/seasonality), **ARIMA/Prophet** baselines, **seq2seq**/**Temporal Fusion Transformers** for complex; **backtesting** with **time-based CV**; **uncertainty** intervals.

## Q17 Spam detection
**Text features** + **behavioral** signals; **Naive Bayes/SVM** historical; **neural** now; **adversarial** spelling attacks—**hybrid** rules.

## Q18 Image classification
**Data** collection/labeling, **augmentation**, **pretrained** CNN/ViT, **calibration**, **monitoring** **slice** performance, **edge** deployment constraints.

## Q19 Sentiment
**Lexicon** baselines, **fine-tuned** BERT, **multilingual** needs, **sarcasm** challenge—**human eval**.

## Q20 Churn
**Survival** analysis, **lead time** for interventions, **class imbalance**, **uplift** modeling for **treatment** effect.

## Q21 Ranking search
**LTR** datasets, **NDCG**, **position bias** **IPW**, **two-tower** retrieval + **cross-encoder** rerank.

## Q22 Network anomaly
**High-dimensional** streams, **unsupervised** (isolation forest, autoencoders), **alert fatigue** control.

### One-line Revision
Each applied problem needs domain metrics, leakage-safe validation, and monitoring—match model family to data and SLOs.

### Difficulty Tag
🟡 Medium

---

# Q23: How do you choose the right machine learning algorithm?

## 1. 🔹 Direct Answer
**Start** with **baseline** (linear, GBDT, small NN) given **data size**, **structure** (tabular/text/image), **latency**, **interpretability**. **Iterate** with **error analysis**. **No** universal winner—**empirical** comparison with **proper CV**.

## 2. 🔹 Intuition
**Occam**: simplest **meeting** SLO wins for **maintainability**.

## 3. 🔹 Deep Dive
**Tabular**: **XGBoost** strong; **deep** if huge data/unstructured.

## 4. 🔹 Practical Perspective
**Product** constraints (explainability) matter.

## 5. 🔹 Code Snippet
```text
if tabular and n<100k: start sklearn GBM ; if text: transformers
```

## 6. 🔹 Interview Follow-ups
1. **Q:** AutoML? **A:** Good exploration—watch leakage and cost.

## 7. 🔹 Common Mistakes
**BERT** for every tiny tabular task.

## 8. 🔹 Comparison / Connections
No free lunch informally.

## 9. 🔹 One-line Revision
Choose algorithms from data modality, size, latency, and interpretability—validate with strong baselines and CV.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q24: What is model drift, and how do you handle it?

## 1. 🔹 Direct Answer
**Data drift**: input **distribution** changes; **concept drift**: **P(y|x)** changes. **Detect** via **feature** **PSI/KS**, **label** delay analysis, **online** metrics. **Handle**: **retrain** triggers, **warm** starts, **adaptive** calibration, **fallback** models.

## 2. 🔹 Intuition
World **changes**—static model **rots**.

## 3. 🔹 Deep Dive
**Seasonality** vs **sudden** shocks—different **response**.

## 4. 🔹 Practical Perspective
**Automated** retraining pipelines with **approval** gates.

## 5. 🔹 Code Snippet
```text
alert if PSI(feature) > 0.2 or auc drops > threshold
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Label drift? **A:** Calibration shift—monitor **ECE**.

## 7. 🔹 Common Mistakes
**Yearly** retrain on **calendar** not **performance** triggers.

## 8. 🔹 Comparison / Connections
Continual learning.

## 9. 🔹 One-line Revision
Monitor drift with statistical tests and business metrics; retrain and recalibrate on schedule or alerts.

## 10. 🔹 Difficulty Tag
🟣 Hard

---

# Q25–Q27: Large-scale training, noisy data, training time optimization

## Q25 Large-scale data
**Distributed** training (**data parallel**), **sharding**, **incremental** learning, **importance sampling**, **feature stores**, **data validation** (Great Expectations).

## Q26 Noisy labels
**Robust loss**, **label cleaning**, **co-teaching**, **small-loss** trick, **crowd** consensus, **semi-supervised**.

## Q27 Training time
**Mixed precision**, **larger batch** + LR scaling, **gradient accumulation**, **distillation**, **early stopping**, **efficient** architectures, **better** hardware.

### One-line Revision
Scale data and training with distributed systems and data quality; speed up with mixed precision, distillation, and infrastructure.

### Difficulty Tag
🟣 Hard

---

# Q28–Q34: Deploy, monitor, low latency, challenges, scalability, explainability, debugging, fairness

## Q28 Deploy ML in production
**Packaging** (Docker), **model registry**, **CI/CD**, **canary**, **rollback**, **API** **(REST/gRPC)**, **batch vs online**.

## Q29 Monitor performance
**Data** quality, **latency**, **throughput**, **errors**, **drift**, **business KPIs**, **SLIs/SLOs**, **dashboards**, **pager** thresholds.

## Q30 Low-latency deployment
**Distillation**, **quantization**, **caching**, **approximate** retrieval, **async** where possible, **regional** edge, **precompute**.

## Q31 Deployment challenges
**Train-serve skew**, **schema** changes, **versioning**, **dependencies**, **security**, **GPU** availability.

## Q32 Scalability & latency requirements
**Horizontal** scaling, **load balancing**, **autoscaling**, **async** queues, **partitioning**.

## Q33 Model explainability
**Why important**: **trust**, **compliance**, **debugging**. **SHAP**, **LIME**, **attention** maps, **counterfactuals**—**trade** fidelity vs cost.

## Q34 Interpretability techniques
**Linear** models, **trees**, **permutation** importance, **partial dependence**, **rule extraction**, **prototypes**.

## Q35 Debugging underperformance
**Slice** analysis, **data** checks, **leakage** hunt, **ablations**, **compare** offline/online, **replay** bad cases.

## Q36 Fairness in production
**Slice** metrics, **bias** **mitigation** in training, **monitoring** cohorts, **appeals**.

### One-line Revision
Production ML = reliable deploy + observability + governance—debug with slices and align offline/online.

### Difficulty Tag
🟣 Hard

---

# Q37: Explain MLOps and its key components.

## 1. 🔹 Direct Answer
**MLOps** applies **DevOps** practices to ML: **versioning** (data, code, models), **CI/CD** for **pipelines**, **experiment tracking**, **model registry**, **monitoring**, **governance**, **reproducibility**.

## 2. 🔹 Intuition
**Ship models** **reliably** **like** software—but **data** adds **entropy**.

## 3. 🔹 Deep Dive
**Testing**: **unit** on code, **validation** on data schemas, **model** **cards**.

## 4. 🔹 Practical Perspective
Tools: **MLflow**, **Kubeflow**, **Vertex**, **SageMaker**.

## 5. 🔹 Code Snippet
```text
git + dvc + mlflow + ci pipeline + k8s deploy
```

## 6. 🔹 Interview Follow-ups
1. **Q:** vs DataOps? **A:** Overlap—feature pipelines critical.

## 7. 🔹 Common Mistakes
**Manual** notebook deploys without tests.

## 8. 🔹 Comparison / Connections
LLMOps, platform engineering.

## 9. 🔹 One-line Revision
MLOps standardizes building, deploying, and monitoring ML with versioning and automation.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q38: What is a feature store, and why is it important?

## 1. 🔹 Direct Answer
**Centralized** **serving** and **training** **features** with **point-in-time** **correctness**—prevents **train-serve skew** and **leakage**. **Offline** store for training, **online** **low-latency** lookup for inference.

## 2. 🔹 Intuition
**Single source of truth** for **feature definitions**.

## 3. 🔹 Deep Dive
**Feast**, **Tecton**, **Databricks** Feature Engineering—**entity** keyed.

## 4. 🔹 Practical Perspective
**Cost** vs **duplicated** **featurization** in silos.

## 5. 🔹 Code Snippet
```text
get_features(user_id, as_of=t) -> consistent in train & serve
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Point-in-time joins? **A:** Only use data available at decision time.

## 7. 🔹 Common Mistakes
**Training** on **future** aggregates without time travel.

## 8. 🔹 Comparison / Connections
Data warehouse, RAG document stores analog.

## 9. 🔹 One-line Revision
Feature store provides consistent, time-correct features for training and serving—reduces skew and leakage.

## 10. 🔹 Difficulty Tag
🟣 Hard

---

# Q39: Cloud vs on-device model deployment.

## 1. 🔹 Direct Answer
**Cloud**: **large** models, **easy updates**, **centralized** monitoring—needs **network**, **latency**, **privacy** trade-offs. **On-device**: **privacy**, **low latency**, **offline**—**constrained** **memory/compute**, **harder** updates—**quantization**, **CoreML/TFLite**.

## 2. 🔹 Intuition
**Horsepower** vs **pocket** constraints.

## 3. 🔹 Deep Dive
**Federated learning** hybrid for privacy.

## 4. 🔹 Practical Perspective
**Hybrid**: **small** on-device + **cloud** fallback.

## 5. 🔹 Code Snippet
```text
edge: int8 model ; cloud: full model + logging
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Update on device? **A:** OTA packages, **versioning**, **A/B** difficult.

## 7. 🔹 Common Mistakes
Ignoring **thermal** **battery** limits mobile.

## 8. 🔹 Comparison / Connections
Edge AI, split inference.

## 9. 🔹 One-line Revision
Cloud for power and iteration; on-device for privacy and latency—often hybrid with compressed models.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q40: Model compression techniques.

## 1. 🔹 Direct Answer
**Quantization** (INT8/INT4), **pruning** (magnitude, structured), **distillation** (teacher-student), **low-rank** factorization, **knowledge** transfer, **architecture** search for **tiny** nets.

## 2. 🔹 Intuition
**Shrink** **compute** and **memory** with **acceptable** accuracy drop.

## 3. 🔹 Deep Dive
**PTQ vs QAT**; **N:M** sparsity on **Ampere**.

## 4. 🔹 Practical Perspective
**Measure** **latency** not only **size**.

## 5. 🔹 Code Snippet
```python
torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Lottery ticket? **A:** Sparse subnetworks trainable from init—researchy.

## 7. 🔹 Common Mistakes
**Pruning** without **fine-tune** recovery.

## 8. 🔹 Comparison / Connections
NAS, MoE for efficiency.

## 9. 🔹 One-line Revision
Compress via quantize, prune, distill, and efficient architectures—validate on-task latency and quality.

## 10. 🔹 Difficulty Tag
🟣 Hard

---

# Q41: Scalability and latency (summary integration)

## 1. 🔹 Direct Answer
**Scale** **horizontally** **stateless** **inference** workers; **cache** **hot** **features**; **batch** **requests** where possible; **partition** **data**; **CDN** for **static**; **GPU** **autoscaling** with **warm** pools; **async** **pipelines** for **non-interactive**.

## 2. 🔹 Intuition
**Remove bottlenecks** **measured** in **profiling**—often **I/O** and **feature** **retrieval**, not **matmul**.

## 3. 🔹 Deep Dive
**Tail latency** **SLOs**—**hedging**, **degradation** **modes**.

## 4. 🔹 Practical Perspective
**Load test** with **realistic** **distributions**.

## 5. 🔹 Code Snippet
```text
p99 latency budget: network + featurizer + model + postprocess
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Multi-region? **A:** **Consistency** vs **latency** for **models**.

## 7. 🔹 Common Mistakes
Optimizing **model** while **DB** is **slow**.

## 8. 🔹 Comparison / Connections
SRE practices, capacity planning.

## 9. 🔹 One-line Revision
Scale ML systems with horizontal serving, caching, efficient features, and rigorous latency profiling—not only model FLOPs.

## 10. 🔹 Difficulty Tag
🟣 Hard

---
