# System Design and MLOps

These answers are designed for ML system design rounds, where interviewers care less about naming models and more about problem framing, data flow, serving architecture, metrics, and operational tradeoffs.

---

# Q1-Q13: End-to-end ML system design patterns (recommendation, search, feed, safety, multimodal, ads, delivery, image search, friends, e-commerce rec)

**Interview-ready answer**

For almost any large-scale ML system, I use the same structure. First I clarify the objective, constraints, and failure costs: what business metric matters, what latency and freshness targets exist, and what harms must be controlled. Then I design the data and feature pipeline, decide how labels are generated, and build a serving architecture that usually separates candidate generation from heavier ranking or decision logic. Finally, I define offline evaluation, online experimentation, fallback behavior, and monitoring for drift, quality, and business KPIs.

**Reusable design skeleton**

- Clarify the task: ranking, retrieval, classification, forecasting, moderation, or matching
- Define scale: QPS, latency SLO, freshness, number of users/items/documents
- Build data pipelines and feature storage with train-serving consistency
- Choose a multi-stage architecture when the candidate space is large
- Evaluate offline with the right metric, then validate online with experiments
- Plan cold start, abuse resistance, privacy, and rollback paths up front

**Strong interview nuance**

In recommendation and search, retrieval and ranking are usually separate because the candidate space is too large for a single heavy model. In high-risk domains such as safety, fraud, or trust, the system often needs rules, human review, and escalation policies in addition to ML.

---

# Q14: How would you build a system to detect fraudulent transactions?

**Interview-ready answer**

Fraud detection is a low-latency, highly imbalanced, adversarial classification problem. I would design it as a layered system: deterministic rules for obvious blocks, a real-time ML scorer for nuanced risk, and a human review queue for uncertain high-value cases. Features would include transaction amount, merchant metadata, velocity features, device or graph signals, historical behavior, and location mismatch indicators. I would optimize for business cost, not accuracy, because false positives hurt users while false negatives lose money.

**What strong candidates mention**

- Real-time feature freshness and train-serving consistency
- Label delay and feedback loops, since confirmed fraud often arrives later
- Precision at top risk thresholds, loss prevented, and review efficiency
- Continuous retraining and monitoring because attackers adapt

---

# Q15: Multimodal Fusion - Early vs Late Fusion.

**Interview-ready answer**

Early fusion combines modalities before or during representation learning so the model can learn cross-modal interactions directly. Late fusion keeps modality-specific models separate and combines their outputs later, which is simpler and more modular. Early fusion can be more powerful when interactions between modalities matter, but it is harder to train and less robust to missing modalities. Late fusion is easier to debug and update, but it may miss deeper cross-modal structure.

**Rule of thumb**

- Early or cross-attention fusion when interaction is central
- Late fusion when engineering simplicity, modularity, or missing data handling matters more

---

# Q16-Q22: Applied ML scenarios (time series, spam, image classification, sentiment, churn, ranking, anomaly traffic, algorithm choice)

**Interview-ready answer**

Across applied scenarios, the correct model choice always follows the same logic: understand the data shape, evaluation metric, feedback loop, and deployment constraints before talking architecture. For time series I would focus on leakage-safe backtesting and horizon-specific error. For spam and abuse I would expect adversarial behavior and combine text features with metadata or rules. For churn and ranking I would define the business action clearly, because predicting an outcome is less useful than enabling the right intervention or ordering. For anomaly detection I would be explicit about alert fatigue, label scarcity, and what "normal" means operationally.

**How to sound senior**

Say that the winning model is rarely the most complex one; it is the one that fits the data, objective, and operational constraints best.

---

# Q23: How do you choose the right machine learning algorithm?

**Interview-ready answer**

I choose algorithms based on data modality, data volume, latency budget, interpretability requirements, and the amount of engineering overhead the team can support. For tabular data, strong baselines like linear models or gradient-boosted trees often win. For text, images, audio, and other unstructured data, pretrained deep models are often the right starting point. The most important principle is to begin with a credible baseline, then justify complexity only when it clearly improves the metric that matters under the deployment constraints.

---

# Q24: What is model drift, and how do you handle it?

**Interview-ready answer**

Model drift is the degradation of performance over time because the environment changes. That can happen because the input distribution changes, the relationship between inputs and labels changes, or the label distribution itself shifts. I handle drift by monitoring input features, prediction distributions, delayed label-based metrics, calibration, and slice performance. Then I define retraining triggers, recalibration strategies, and fallback behavior rather than waiting for a major incident.

**Good nuance**

Not every distribution shift is harmful. The important question is whether the shift changes decision quality.

---

# Q25-Q27: Large-scale training, noisy data, training time optimization

**Interview-ready answer**

For large-scale training, I think in terms of data quality, system throughput, and cost efficiency together. Scale problems are often solved through better pipelines, sharding, mixed precision, caching, and distributed training, not only by adding GPUs. For noisy data, I would combine better data validation with robust objectives, relabeling workflows, or confidence-based filtering. For training time, I optimize the bottleneck first: data loading, sequence length, batch formation, model size, or communication overhead.

**Good interview framing**

If training is slow, the answer is not automatically "bigger hardware." It is usually "find the dominant bottleneck and remove wasted work."

---

# Q28-Q34: Deploy, monitor, low latency, challenges, scalability, explainability, debugging, fairness

**Interview-ready answer**

Production ML is about turning a model into a reliable system. That means packaging and versioning, reproducible feature computation, deployment safety through canaries or shadow mode, monitoring latency and data quality, debugging prediction failures, and building enough explainability for operators and stakeholders. Low latency often requires architecture choices such as candidate retrieval, caching, batching, compression, or distillation. Fairness and explainability are not add-ons; they influence what features you use, how you validate, and what safeguards the system needs.

**What strong candidates add**

- Always define fallback behavior if the model or feature service fails
- Monitor both technical metrics and business outcomes
- Prefer progressive rollout over big-bang deployment
- Debugging often starts with feature logs and cohort-level analysis

---

# Q37: Explain MLOps and its key components.

**Interview-ready answer**

MLOps is the discipline of making ML systems reproducible, deployable, observable, and maintainable across their lifecycle. The core components are data and model versioning, pipeline orchestration, feature management, experiment tracking, validation and testing, model registry, deployment automation, and monitoring after launch. The deeper point is that MLOps exists because ML systems fail in more ways than normal software: data shifts, labels arrive late, features skew between training and serving, and model quality decays even when the code has not changed.

---

# Q38: What is a feature store, and why is it important?

**Interview-ready answer**

A feature store is a system for defining, storing, and serving features consistently across training and inference. Its importance is that it reduces train-serving skew, improves reuse, and creates a governed place for feature definitions, freshness, lineage, and access patterns. In interviews, say that a feature store is not just a database of columns; it is a contract that helps keep ML features correct and reproducible across pipelines.

---

# Q39: Cloud vs on-device model deployment.

**Interview-ready answer**

Cloud deployment gives you more compute, easier updates, central monitoring, and access to richer features, but it adds latency, network dependence, and privacy considerations. On-device deployment offers low latency, offline capability, and better privacy, but it is constrained by memory, battery, compute, and update complexity. The right answer depends on whether the bottleneck is responsiveness, privacy, bandwidth, cost, or model size.

---

# Q40: Model compression techniques.

**Interview-ready answer**

Model compression reduces memory footprint and inference cost while trying to preserve accuracy. The common techniques are quantization, pruning, distillation, low-rank factorization, and architecture redesign. The best answer in an interview is not a list but a tradeoff statement: compression is valuable when serving cost or latency is the constraint, but every technique introduces some accuracy-risk and engineering complexity.

---

# Q41: Scalability and latency (summary integration)

**Interview-ready answer**

Scalability is about handling growth in traffic, data volume, and model complexity without losing reliability or cost control. Latency is about meeting the response-time budget for a single request or batch. In ML systems, the two are tightly linked: the architecture that scales well is often the one that keeps expensive computation off the critical path through precomputation, retrieval stages, caching, batching, and lightweight online models. A strong closing line is that system design is ultimately about deciding what should be computed offline, online, and not at all.
