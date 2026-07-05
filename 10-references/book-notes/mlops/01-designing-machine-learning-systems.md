---
module: References
topic: Book Notes
subtopic: Mlops Designing Machine Learning Systems
status: unread
tags: [references, ml, book-notes-mlops]
---
# Designing Machine Learning Systems

## Chapter 1: When to Use ML and ML in Production vs Research

**The problem the book is addressing**
Teams spend months building ML systems for problems that don't require ML, or fail to account for the structural differences between research and production. A system that achieves high accuracy in a Jupyter notebook may be unusable in production due to latency, maintenance burden, or shifting data.

**The core insight**
ML is appropriate when: the problem requires learning patterns from data (not rules), there is sufficient data available, the pattern is complex enough to justify the engineering overhead, and the cost of being wrong is acceptable. ML in production adds constraints that research ignores: multiple stakeholders, latency requirements, shifting data distributions, fairness, and interpretability.

**The mechanics**
- ML is the right tool when: features are hard to engineer manually, data is abundant (millions of examples), patterns change over time (static rules go stale)
- Production ML constraints: model accuracy (ML team), inference latency (engineering), fairness/bias (legal/compliance), explainability (regulators/users)
- ML vs traditional software: behavior is encoded in data + model, not code; bugs are often statistical, not deterministic; debugging requires different tools
- Data versioning: unlike code, data changes — you need to know which training data produced which model
- Data poisoning: adversarial inputs can corrupt model behavior — important for security-sensitive applications

**What the book gets right / what to watch out for**
The research-vs-production distinction is the book's most important contribution — most ML courses only teach research workflows. The data versioning requirement is often ignored until a production incident makes it critical. Fairness requirements differ by jurisdiction and application domain — this is not a one-time check but an ongoing monitoring requirement.

---

## Chapter 2: ML Systems Design — Framing and Requirements

**The problem the book is addressing**
Poorly framed ML problems fail regardless of model quality. A team can spend months building a model that optimizes the wrong objective, solves the wrong subproblem, or ignores scalability and reliability requirements that make it impossible to deploy.

**The core insight**
Before writing any code, define: the business objective, the ML proxy objective (what you can measure), reliability requirements (acceptable downtime, latency SLAs), scalability targets (peak QPS), maintainability (who will own this), and adaptability (how quickly must the model respond to distribution shift). The ML problem formulation determines the entire downstream pipeline.

**The mechanics**
- Business objective → ML objective: translate "increase revenue" to "increase CTR" to "binary classification on user-item pairs"
- Task types: binary classification, multiclass, multilabel, regression, ranking, structured prediction
- Reliability: what happens when the model fails? Graceful degradation (fallback to rule-based system) is usually required
- Scalability: can the system handle 10× traffic? Online prediction scales differently from batch
- Iterative process: data collection → feature engineering → model training → evaluation → deployment → monitoring → data collection

**What the book gets right / what to watch out for**
The framing-before-modeling discipline is the most important habit for senior ML practitioners. The "mind vs data" debate — whether more data or better algorithms matters more — is context-dependent: for well-defined tasks with clear objectives, data quantity usually wins; for novel tasks, inductive biases and architecture matter more.

---

## Chapter 3: Data Engineering — Formats, Storage, and Processing

**The problem the book is addressing**
Data engineering decisions made early in a project are expensive to change later. The wrong storage format, data model, or processing architecture creates bottlenecks that limit model iteration speed.

**The core insight**
Row-major formats (CSV, JSON) are optimized for transactional access (one record at a time). Column-major formats (Parquet, ORC) are optimized for analytical access (all values of one feature across many records). ML feature access is column-major — use Parquet. OLTP databases optimize for writes; OLAP systems optimize for analytical reads — don't run ML pipelines on production OLTP databases.

**The mechanics**
- Row-major: CSV, JSON — easy to read/write, bad for ML (read entire row to access one column)
- Column-major: Parquet, ORC — compress each column independently; fast analytical reads; standard for ML feature pipelines
- Data models: relational (SQL, ACID transactions) for structured data; document (MongoDB) for flexible schemas; time-series (InfluxDB) for telemetry
- OLTP vs OLAP: OLTP = transactional (insert/update/delete); OLAP = analytical (aggregate queries, reporting)
- ETL vs ELT: Extract-Transform-Load (transform before storage) vs Extract-Load-Transform (transform in data warehouse); ELT is now standard with cheap storage
- Dataflow modes: online (trigger on each event), batch (scheduled), stream (continuous but not per-event)
- Batch processing: high throughput, high latency; MapReduce/Spark
- Stream processing: low latency; Kafka + Flink/Spark Streaming

**What the book gets right / what to watch out for**
The Parquet recommendation for ML feature storage is correct and underappreciated — CSV reads of multi-gigabyte files are a major iteration bottleneck. The distinction between OLTP and OLAP is important: running heavy analytical queries on a production database degrades application performance. Separate the analytical workload onto a dedicated warehouse (Snowflake, BigQuery, Redshift).

---

## Chapter 4: Training Data — Sampling, Labeling, and Class Imbalance

**The problem the book is addressing**
Training data quality determines model quality. Practitioners often treat data collection as a one-time task, underinvest in labeling quality, and ignore the statistical biases introduced by their sampling strategy.

**The core insight**
The sampling strategy determines what distribution the model learns. Non-probability sampling (convenience sampling) introduces systematic biases that don't reflect deployment distribution. Class imbalance is a consequence of sampling from reality — it must be explicitly handled because models trained on imbalanced data learn biased decision boundaries.

**The mechanics**
- Sampling strategies: simple random (unbiased, may undersample rare events), stratified (preserves class distribution), weighted (intentionally oversample rare events), reservoir (streaming data), importance (reweight samples to match target distribution)
- Labeling: hand labels (expensive, gold standard), natural labels (user behavior: clicks, purchases), weak supervision (Snorkel — label functions), semi-supervised (use model predictions on unlabeled data), active learning (query most uncertain examples)
- Class imbalance: resampling (oversample minority, undersample majority), cost-sensitive loss (weight minority class loss by n_majority/n_minority), threshold tuning, anomaly detection for extreme imbalance
- Data augmentation: geometric + intensity transforms for images; back-translation for text; Mixup/CutMix for both

**What the book gets right / what to watch out for**
Natural labels (implicit supervision from user behavior) are the most scalable labeling approach — clicks, purchases, and completions provide training signal without manual annotation. The lag between user action and label availability is a key system design challenge. Weak supervision (Snorkel) is underused and powerful — hand-crafted label functions are often 80% as good as manual labeling at 1% of the cost.

---

## Chapter 5: Feature Engineering — Scaling, Encoding, and Leakage

**The problem the book is addressing**
Feature engineering mistakes produce models that appear to work in development but fail in production. Data leakage — where future information contaminates features — is the most common cause of optimistic offline metrics that don't translate to production performance.

**The core insight**
Learned features (embeddings from deep networks) scale better than engineered features, but require more data. Engineered features are more interpretable and require less data. The two approaches complement each other. Data leakage is the critical bug to prevent — any feature created using information unavailable at inference time will produce artificially good offline metrics.

**The mechanics**
- Handling missing values: imputation (mean/median/model-based), flag as a separate category (for categoricals), indicator column (model learns from the fact of missingness)
- Scaling: min-max normalization → [0,1]; standardization → zero mean unit variance; log transform for skewed distributions
- Discretization: binning continuous → categorical; uniform bins, quantile bins, or k-means bins
- Encoding: one-hot for low-cardinality; embedding lookup for high-cardinality; hashing trick for very high cardinality
- Feature crossing: pairwise product of features; adds explicit nonlinearity; must be done identically at train and serve
- Feature importance: SHAP, permutation importance — remove features that don't improve performance
- Data leakage detection: check if feature values correlate suspiciously with the label; check that feature computation uses only data available at prediction time

**What the book gets right / what to watch out for**
Feature leakage prevention is the most important topic in this chapter. Common leakage bugs: using a feature computed on the entire dataset before splitting (target encoding without cross-validation), using information from the future in time series features (using tomorrow's data to predict today's outcome). The fix: always compute features inside a cross-validation fold, using only the training data.

---

## Chapter 6: Model Development — Selection, Training, and Evaluation

**The problem the book is addressing**
Model selection without a systematic process produces inconsistent results — the model that performs best in one experiment may be chosen for reasons that don't generalize. Evaluation without careful design produces optimistic metrics that don't reflect production performance.

**The core insight**
Model selection should start from simple baselines (majority class, mean prediction, linear model) before adding complexity. Each step up in complexity requires justification — the performance gain must exceed the maintenance and compute cost. Offline evaluation should be designed to reflect online performance; discrepancies between offline and online metrics indicate a problem.

**The mechanics**
- Model selection: random forest and gradient boosting for tabular; fine-tuned transformers for text; CNNs or ViT for images
- Experiment tracking: MLflow, Weights & Biases, DVC — log hyperparameters, metrics, artifacts, and data versions for every run
- Distributed training: data parallelism (each GPU processes different batches), model parallelism (layers on different GPUs), pipeline parallelism (stages on different GPUs)
- Baselines: majority class, mean prediction, simple rule-based, logistic regression — beat these before justifying complexity
- AutoML: automated pipeline search; useful for baseline establishment; rarely produces optimal results for production
- Offline evaluation: cross-validation for standard problems; held-out temporal test set for time series
- Perturbation tests: model predictions should be robust to small input perturbations
- Invariance tests: predictions should be invariant to irrelevant features (name, demographic in medical diagnosis)
- Calibration: model probabilities should match actual frequencies; evaluate with reliability diagrams; fix with Platt scaling or isotonic regression
- Slice-based evaluation: measure performance on subgroups (demographics, geographic regions, product categories) — aggregate metrics can hide subgroup failures
- Ensembles: combine multiple models; bagging (average predictions of models trained on bootstrap samples), boosting (sequential models focused on errors), stacking (use model predictions as features for a meta-learner)

**What the book gets right / what to watch out for**
Slice-based evaluation is the most underused evaluation technique in industry — aggregate metrics hide systematic failures on important subpopulations. Calibration is particularly important for models that feed downstream decision systems (credit scoring, medical screening) — uncalibrated probabilities produce suboptimal decisions. Experiment tracking from day one is the most impactful habit change for ML teams.

---

## Chapter 7: Deployment — Online vs Batch, Model Compression, Edge

**The problem the book is addressing**
The deployment architecture determines latency, cost, and what optimizations are available. A model deployed synchronously for online prediction has different constraints than one running in a nightly batch job. Practitioners need to match deployment mode to use case requirements.

**The core insight**
Online prediction (synchronous) responds to requests in real time — latency is a first-class constraint. Batch prediction (asynchronous) generates predictions on a schedule — throughput is the constraint. Hybrid approaches pre-compute predictions for common requests and fall back to online for others. Model compression (quantization, distillation, pruning) reduces inference cost at the expense of accuracy.

**The mechanics**
- Online prediction: model server (TorchServe, TF Serving, Triton) receives request; runs inference; returns response; p99 latency < 100ms typical
- Batch prediction: Spark/Beam job processes entire user base overnight; store predictions in database; serve from database at query time
- Hybrid: pre-compute batch predictions for known entities; use online prediction for new entities
- Model compression:
  - Low-rank factorization: decompose weight matrices into products of smaller matrices
  - Knowledge distillation: train small "student" model to match outputs of large "teacher" model
  - Pruning: zero out small weights; structured pruning (remove entire neurons) vs unstructured; retrain after pruning
  - Quantization: reduce weight precision from FP32 → INT8 → INT4; post-training quantization (no retraining) or quantization-aware training
- Edge deployment: run on device (mobile, embedded); requires quantization + compilation for target hardware (CoreML, TFLite, ONNX Runtime)
- Cloud vs edge tradeoffs: cloud has more compute, stale models; edge has lower latency, data privacy, works offline

**What the book gets right / what to watch out for**
The batch-vs-online framing is the correct first design decision for any serving system. Knowledge distillation is the most powerful compression technique for large models — a 10× smaller model can often achieve 95%+ of the teacher's accuracy. Quantization to INT8 produces 4× memory reduction with ~1–2% accuracy loss on most models, making it the default first compression step.

---

## Chapter 8: Data Distribution Shifts — Detecting and Responding

**The problem the book is addressing**
Models degrade after deployment because the world changes. User behavior shifts, products are updated, the economy changes — the data distribution at inference time diverges from training time. Without monitoring, models silently degrade until failures are noticed via business metrics.

**The core insight**
Distribution shift has three types: covariate shift (input distribution P(X) changes but P(Y|X) is stable), label shift (P(Y) changes), concept drift (P(Y|X) itself changes — the most dangerous). Detection requires monitoring feature distributions, output distributions, and model performance simultaneously.

**The mechanics**
- Covariate shift: compare feature distributions at training vs serving time using KS test, PSI (Population Stability Index), or MMD
- Label shift: monitor prediction distribution vs expected label distribution; compare with calibration checks
- Concept drift: most dangerous; hardest to detect without ground truth labels; use proxy metrics or delayed labels
- Feature changes: new categories appear, feature suddenly becomes null, value range expands
- Label schema changes: class definitions change (new fraud patterns, new product categories)
- Monitoring: log every prediction with timestamp and input features; compare recent distribution to reference window
- Response: retrain on recent data, update reference distribution, add new data collection

**What the book gets right / what to watch out for**
The three-way taxonomy of shift types is the most useful framework in this chapter. PSI is the standard industry metric for covariate shift monitoring (PSI < 0.1 = no change, 0.1–0.25 = slight change, > 0.25 = significant shift). Detecting concept drift without labels requires indirect signals — sudden changes in user behavior or business metrics often precede detectable distributional shifts.

---

## Chapter 9: Continual Learning — Retraining Strategies and Safe Deployment

**The problem the book is addressing**
Models decay over time as data distributions shift. Periodic retraining helps but introduces new risks — a newly trained model might perform worse than the current production model, or a bug in the training pipeline could corrupt the new model. Deployment must be safe.

**The core insight**
Continual learning is stateless (retrain from scratch on new data) or stateful (fine-tune from the current model). Stateful is faster but risks catastrophic forgetting. Safe deployment requires testing the new model against the current one before routing user traffic. Bandit-based routing allocates traffic adaptively based on observed performance.

**The mechanics**
- Four stages of continual learning: (1) manual retraining on schedule, (2) automated retraining on schedule, (3) automated retraining on trigger (metric drop, data drift), (4) online learning (update on each example)
- Catastrophic forgetting: fine-tuning on new data causes model to forget old patterns; mitigated by including old data in retraining
- Shadow deployment: new model runs in parallel, predictions logged but not shown to users; compare offline against production model
- A/B testing: split traffic between current and new model; measure business metrics; requires sufficient traffic for statistical significance
- Canary deployment: route small percentage (1–5%) of traffic to new model; monitor for errors; gradually increase if no issues
- Interleaving experiments: show results from both models in same session (eliminates user variance); used by search/recommendation systems
- Bandits: Thompson sampling or UCB allocates more traffic to better-performing model dynamically; faster convergence than A/B test

**What the book gets right / what to watch out for**
Shadow deployment is the safest way to validate a new model — it catches bugs and distribution mismatches before users are affected. Canary deployment is the standard first step before full rollout. Bandits are underused in industry — they are strictly better than A/B testing when you have a clear online metric and sufficient traffic to estimate it accurately.

---

## Chapter 10: Infrastructure — Storage, Compute, Platforms

**The problem the book is addressing**
ML infrastructure choices determine iteration speed. Teams that cobble together scripts on shared servers take 10× longer to run experiments and deploy models than teams with proper ML platforms. But over-engineering infrastructure before establishing product-market fit wastes engineering resources.

**The core insight**
ML infrastructure has four layers: storage (raw data, features, models, artifacts), compute (training, inference), development environment (notebooks, experiment tracking), and orchestration (scheduling, dependency management). Each layer has build vs buy tradeoffs that depend on team size and problem complexity.

**The mechanics**
- Storage: object storage (S3/GCS) for raw data and model artifacts; feature store (Feast, Tecton) for serving consistent features; model store (MLflow, W&B) for versioned models
- Compute: spot/preemptible instances for training (cheap but interruptible); dedicated instances for serving (predictable latency)
- Development: Jupyter Notebooks for exploration; experiment tracking (MLflow, W&B, Neptune) for reproducibility; DVC for data versioning
- Orchestration: Airflow (general DAG scheduler), Kubeflow (Kubernetes-native ML pipelines), Metaflow (data science workflows); use Kubeflow for complex GPU pipelines, Airflow for simpler ETL
- ML platforms: SageMaker (AWS-integrated, managed), Vertex AI (GCP-integrated), Azure ML; full-service platforms reduce infra overhead but create vendor lock-in
- Build vs buy: buy infrastructure (cloud compute, storage, orchestration); build features specific to your domain; don't build what cloud vendors already provide

**What the book gets right / what to watch out for**
The feature store concept is often underappreciated until teams hit the training-serving skew problem — the same feature is computed differently in batch training and online serving, causing silent model degradation. A feature store enforces a single definition. The build vs buy guidance is practical: start with managed services, build custom solutions only when vendor limitations are a demonstrable bottleneck.

---

## Chapter 11: The Human Side — UX, Ethics, and Team Structure

**The problem the book is addressing**
ML systems affect real users and communities. A model that is technically correct but produces inconsistent, unexplainable, or biased predictions creates user harm and erodes trust. Teams that treat fairness and UX as afterthoughts discover the problem after deployment when the cost of fixing it is much higher.

**The core insight**
"Mostly correct" predictions are worse than no predictions for high-stakes decisions — a medical model that is wrong 5% of the time is not helpful if users trust it 100% of the time. Smooth failing (fallback to a safe default when confidence is low) is better than confident wrong predictions. Responsible AI is a cross-functional responsibility, not an ML team add-on.

**The mechanics**
- UX inconsistency: different predictions for the same user on consecutive requests destroy trust; cache or deterministically seed predictions for consistency
- Smooth failing: when model confidence is below a threshold, fall back to a rule-based system or present multiple options
- Bias identification: measure performance metrics by demographic group, geographic region, and other relevant slices; use AI Fairness 360 (IBM) for statistical tests
- Bias mitigation: pre-processing (rebalance training data), in-processing (fairness constraints during training), post-processing (adjust decision thresholds per group)
- Cross-functional teams: ML engineers + data engineers + domain experts + legal/compliance + UX — decisions that affect any of these domains need their input
- Responsible AI checklist: data collection consent, demographic representation, disparate impact analysis, explainability for regulated decisions, monitoring for bias drift

**What the book gets right / what to watch out for**
The inconsistency problem is one of the most overlooked UX issues in ML systems — users quickly notice when the same query returns different results, and it damages credibility. Bias mitigation is more effective pre-training (fix the data) than post-training (adjust thresholds) — post-processing adjustments can satisfy statistical fairness criteria while the model still encodes biased patterns.

## Flashcards

**ML is the right tool when?** #flashcard
features are hard to engineer manually, data is abundant (millions of examples), patterns change over time (static rules go stale)

**Production ML constraints?** #flashcard
model accuracy (ML team), inference latency (engineering), fairness/bias (legal/compliance), explainability (regulators/users)

**ML vs traditional software?** #flashcard
behavior is encoded in data + model, not code; bugs are often statistical, not deterministic; debugging requires different tools

**Data versioning: unlike code, data changes?** #flashcard
you need to know which training data produced which model

**Data poisoning: adversarial inputs can corrupt model behavior?** #flashcard
important for security-sensitive applications

**Business objective → ML objective?** #flashcard
translate "increase revenue" to "increase CTR" to "binary classification on user-item pairs"

**Task types?** #flashcard
binary classification, multiclass, multilabel, regression, ranking, structured prediction

**Reliability?** #flashcard
what happens when the model fails? Graceful degradation (fallback to rule-based system) is usually required

**Scalability?** #flashcard
can the system handle 10× traffic? Online prediction scales differently from batch

**Iterative process?** #flashcard
data collection → feature engineering → model training → evaluation → deployment → monitoring → data collection

**Row-major: CSV, JSON?** #flashcard
easy to read/write, bad for ML (read entire row to access one column)

**Column-major: Parquet, ORC?** #flashcard
compress each column independently; fast analytical reads; standard for ML feature pipelines

**Data models?** #flashcard
relational (SQL, ACID transactions) for structured data; document (MongoDB) for flexible schemas; time-series (InfluxDB) for telemetry

**OLTP vs OLAP?** #flashcard
OLTP = transactional (insert/update/delete); OLAP = analytical (aggregate queries, reporting)

**ETL vs ELT?** #flashcard
Extract-Transform-Load (transform before storage) vs Extract-Load-Transform (transform in data warehouse); ELT is now standard with cheap storage

**Dataflow modes?** #flashcard
online (trigger on each event), batch (scheduled), stream (continuous but not per-event)

**Batch processing?** #flashcard
high throughput, high latency; MapReduce/Spark

**Stream processing?** #flashcard
low latency; Kafka + Flink/Spark Streaming

**Sampling strategies?** #flashcard
simple random (unbiased, may undersample rare events), stratified (preserves class distribution), weighted (intentionally oversample rare events), reservoir (streaming data), importance (reweight samples to match target distribution)

**Labeling: hand labels (expensive, gold standard), natural labels (user behavior: clicks, purchases), weak supervision (Snorkel?** #flashcard
label functions), semi-supervised (use model predictions on unlabeled data), active learning (query most uncertain examples)

**Class imbalance?** #flashcard
resampling (oversample minority, undersample majority), cost-sensitive loss (weight minority class loss by n_majority/n_minority), threshold tuning, anomaly detection for extreme imbalance

**Data augmentation?** #flashcard
geometric + intensity transforms for images; back-translation for text; Mixup/CutMix for both

**Handling missing values?** #flashcard
imputation (mean/median/model-based), flag as a separate category (for categoricals), indicator column (model learns from the fact of missingness)

**Scaling?** #flashcard
min-max normalization → [0,1]; standardization → zero mean unit variance; log transform for skewed distributions

**Discretization?** #flashcard
binning continuous → categorical; uniform bins, quantile bins, or k-means bins

**Encoding?** #flashcard
one-hot for low-cardinality; embedding lookup for high-cardinality; hashing trick for very high cardinality

**Feature crossing?** #flashcard
pairwise product of features; adds explicit nonlinearity; must be done identically at train and serve

**Feature importance: SHAP, permutation importance?** #flashcard
remove features that don't improve performance

**Data leakage detection?** #flashcard
check if feature values correlate suspiciously with the label; check that feature computation uses only data available at prediction time

**Model selection?** #flashcard
random forest and gradient boosting for tabular; fine-tuned transformers for text; CNNs or ViT for images

**Experiment tracking: MLflow, Weights & Biases, DVC?** #flashcard
log hyperparameters, metrics, artifacts, and data versions for every run

**Distributed training?** #flashcard
data parallelism (each GPU processes different batches), model parallelism (layers on different GPUs), pipeline parallelism (stages on different GPUs)

**Baselines: majority class, mean prediction, simple rule-based, logistic regression?** #flashcard
beat these before justifying complexity

**AutoML?** #flashcard
automated pipeline search; useful for baseline establishment; rarely produces optimal results for production

**Offline evaluation?** #flashcard
cross-validation for standard problems; held-out temporal test set for time series

**Perturbation tests?** #flashcard
model predictions should be robust to small input perturbations

**Invariance tests?** #flashcard
predictions should be invariant to irrelevant features (name, demographic in medical diagnosis)

**Calibration?** #flashcard
model probabilities should match actual frequencies; evaluate with reliability diagrams; fix with Platt scaling or isotonic regression

**Slice-based evaluation: measure performance on subgroups (demographics, geographic regions, product categories)?** #flashcard
aggregate metrics can hide subgroup failures

**Ensembles?** #flashcard
combine multiple models; bagging (average predictions of models trained on bootstrap samples), boosting (sequential models focused on errors), stacking (use model predictions as features for a meta-learner)

**Online prediction?** #flashcard
model server (TorchServe, TF Serving, Triton) receives request; runs inference; returns response; p99 latency < 100ms typical

**Batch prediction?** #flashcard
Spark/Beam job processes entire user base overnight; store predictions in database; serve from database at query time

**Hybrid?** #flashcard
pre-compute batch predictions for known entities; use online prediction for new entities

**Model compression:?** #flashcard
Model compression:

**Low-rank factorization?** #flashcard
decompose weight matrices into products of smaller matrices

**Knowledge distillation?** #flashcard
train small "student" model to match outputs of large "teacher" model

**Pruning?** #flashcard
zero out small weights; structured pruning (remove entire neurons) vs unstructured; retrain after pruning

**Quantization?** #flashcard
reduce weight precision from FP32 → INT8 → INT4; post-training quantization (no retraining) or quantization-aware training

**Edge deployment?** #flashcard
run on device (mobile, embedded); requires quantization + compilation for target hardware (CoreML, TFLite, ONNX Runtime)

**Cloud vs edge tradeoffs?** #flashcard
cloud has more compute, stale models; edge has lower latency, data privacy, works offline

**Covariate shift?** #flashcard
compare feature distributions at training vs serving time using KS test, PSI (Population Stability Index), or MMD

**Label shift?** #flashcard
monitor prediction distribution vs expected label distribution; compare with calibration checks

**Concept drift?** #flashcard
most dangerous; hardest to detect without ground truth labels; use proxy metrics or delayed labels

**Feature changes?** #flashcard
new categories appear, feature suddenly becomes null, value range expands

**Label schema changes?** #flashcard
class definitions change (new fraud patterns, new product categories)

**Monitoring?** #flashcard
log every prediction with timestamp and input features; compare recent distribution to reference window

**Response?** #flashcard
retrain on recent data, update reference distribution, add new data collection

**Four stages of continual learning?** #flashcard
(1) manual retraining on schedule, (2) automated retraining on schedule, (3) automated retraining on trigger (metric drop, data drift), (4) online learning (update on each example)

**Catastrophic forgetting?** #flashcard
fine-tuning on new data causes model to forget old patterns; mitigated by including old data in retraining

**Shadow deployment?** #flashcard
new model runs in parallel, predictions logged but not shown to users; compare offline against production model

**A/B testing?** #flashcard
split traffic between current and new model; measure business metrics; requires sufficient traffic for statistical significance

**Canary deployment?** #flashcard
route small percentage (1–5%) of traffic to new model; monitor for errors; gradually increase if no issues

**Interleaving experiments?** #flashcard
show results from both models in same session (eliminates user variance); used by search/recommendation systems

**Bandits?** #flashcard
Thompson sampling or UCB allocates more traffic to better-performing model dynamically; faster convergence than A/B test

**Storage?** #flashcard
object storage (S3/GCS) for raw data and model artifacts; feature store (Feast, Tecton) for serving consistent features; model store (MLflow, W&B) for versioned models

**Compute?** #flashcard
spot/preemptible instances for training (cheap but interruptible); dedicated instances for serving (predictable latency)

**Development?** #flashcard
Jupyter Notebooks for exploration; experiment tracking (MLflow, W&B, Neptune) for reproducibility; DVC for data versioning

**Orchestration?** #flashcard
Airflow (general DAG scheduler), Kubeflow (Kubernetes-native ML pipelines), Metaflow (data science workflows); use Kubeflow for complex GPU pipelines, Airflow for simpler ETL

**ML platforms?** #flashcard
SageMaker (AWS-integrated, managed), Vertex AI (GCP-integrated), Azure ML; full-service platforms reduce infra overhead but create vendor lock-in

**Build vs buy?** #flashcard
buy infrastructure (cloud compute, storage, orchestration); build features specific to your domain; don't build what cloud vendors already provide

**UX inconsistency?** #flashcard
different predictions for the same user on consecutive requests destroy trust; cache or deterministically seed predictions for consistency

**Smooth failing?** #flashcard
when model confidence is below a threshold, fall back to a rule-based system or present multiple options

**Bias identification?** #flashcard
measure performance metrics by demographic group, geographic region, and other relevant slices; use AI Fairness 360 (IBM) for statistical tests

**Bias mitigation?** #flashcard
pre-processing (rebalance training data), in-processing (fairness constraints during training), post-processing (adjust decision thresholds per group)

**Cross-functional teams: ML engineers + data engineers + domain experts + legal/compliance + UX?** #flashcard
decisions that affect any of these domains need their input

**Responsible AI checklist?** #flashcard
data collection consent, demographic representation, disparate impact analysis, explainability for regulated decisions, monitoring for bias drift
