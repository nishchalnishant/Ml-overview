---
module: References
topic: Book Notes
subtopic: Mlops Machine Learning Design Patterns
status: unread
tags: [references, ml, book-notes-mlops]
---
# Machine Learning Design Patterns

## Chapter 1: The Need for ML Design Patterns

**The problem the book is addressing**
ML teams repeatedly encounter the same problems — data quality issues, training-serving skew, reproducibility failures, deployment bugs — but solve them ad hoc each time. Without a shared vocabulary and catalog of proven solutions, teams waste time and introduce avoidable errors.

**The core insight**
ML design patterns are reusable solutions to recurring problems in the ML lifecycle, analogous to software design patterns (GoF). Documenting them with names (Hashed Feature, Transform, Shadow Mode) gives teams a shared vocabulary and prevents reinventing the wheel for known problems.

**The mechanics**
- Data quality dimensions: accuracy (correct values), completeness (no missing fields), consistency (same format across systems), timeliness (data available when needed)
- ML lifecycle phases: discovery (problem definition, data exploration) → development (feature engineering, training, evaluation) → deployment (serving, monitoring, maintenance)
- AI readiness levels: tactical (point solutions, individual models), strategic (platform, shared infrastructure), transformational (ML as core business capability)
- Reproducibility: same code + same data + same environment → same result; requires version control for code, data, and dependencies

**What the book gets right / what to watch out for**
The design pattern framing is the book's key contribution — it elevates ML engineering to a discipline with named, documented solutions. The AI readiness levels are a useful framework for organizational assessment. The Google Cloud orientation means some patterns are biased toward GCP tooling (Vertex AI, BigQuery ML) — the underlying concepts transfer to any cloud provider.

---

## Chapter 2: Data Representation Patterns

**The problem the book is addressing**
Raw features are rarely in the form that ML models learn from efficiently. Numeric inputs need scaling. Categorical inputs need encoding. High-cardinality features explode the parameter count. Non-linear relationships require explicit feature construction. Each of these is a recurring problem with a known solution.

**The core insight**
The representation of features determines what relationships the model can capture. A linear model with the right feature transformations can match a complex model on simple relationships. The choice between explicit engineering (feature cross, discretization) and learned representation (embedding) depends on data size and interpretability requirements.

**The mechanics**

### Numerical Inputs and Scaling
- Standardization: (x - μ) / σ → zero mean unit variance; required for distance-based and gradient-sensitive models
- Min-max normalization: (x - x_min) / (x_max - x_min) → [0,1]; use when bounds are known
- Log transform: log(1+x) for right-skewed features; brings outliers closer to the bulk of the distribution

### Hashed Feature Pattern
- Problem: high-cardinality categoricals (user IDs, product SKUs) can't be one-hot encoded — vocabulary too large, cold-start for new values
- Solution: `hash(feature_value) % num_buckets` maps any value to a fixed number of buckets; works for new values at inference
- Trade-off: bucket collisions (two different values map to the same bucket) introduce noise; larger bucket count reduces collisions but increases parameters
- Use when: feature cardinality > 10,000, or new values appear at serving time (cold-start problem)

### Reframing Pattern
- Problem: a continuous output (regression) can be reframed as a classification over buckets; a binary label can be reframed as a regression over probability
- Solution: regression → classification: bucket the output into ranges (e.g., rainfall in mm → {no rain, light, moderate, heavy}); lose ranking within buckets but gain more informative loss signal
- When to reframe: when the distribution of the target is multimodal, or when you need calibrated interval predictions rather than point predictions

### Feature Cross Pattern
- Problem: linear models can't capture interactions between features (user_age × product_category effects)
- Solution: explicitly create a new feature that is the Cartesian product of two categorical features: hour_of_day × day_of_week → 168 combinations
- Trade-off: multiplies feature space size; increases risk of overfitting; most useful for linear models
- Modern alternative: embedding-based models (neural networks) learn feature interactions implicitly

### Multimodal Input Pattern
- Problem: real-world entities are described by multiple data types (product: image + text description + numerical price)
- Solution: separate encoders per modality (CNN for image, BERT for text, FC for numerical); concatenate latent representations; joint fine-tuning
- Late fusion vs early fusion: late fusion (concatenate after encoding) is simpler and handles missing modalities; early fusion (combine raw features) requires all modalities

**What the book gets right / what to watch out for**
The hashed feature pattern is underused and important for production systems that must handle unseen categories at inference time. Feature crosses are most valuable for logistic regression models — neural networks learn multiplicative interactions implicitly. Multimodal models are the current frontier — CLIP, Flamingo, and Gemini demonstrate that joint training across modalities produces representations better than any single modality alone.

---

## Chapter 3: Problem Representation Patterns

**The problem the book is addressing**
The same real-world task can be framed as multiple different ML problems. Choosing the wrong framing produces poor results regardless of model quality. Patterns that reshape the problem itself are often more valuable than patterns that improve the model.

**The core insight**
Problem framing is a design decision, not a given. Regression can become classification (and vice versa). Single-label can become multilabel. A single complex model can be replaced by a cascade of simpler models. The neutral class pattern handles inherent ambiguity that forces the model into incorrect confident predictions.

**The mechanics**

### Rebalancing Pattern
- Problem: class imbalance causes models to optimize for majority class; minority class performance is poor
- Solutions: downsampling majority (discard majority examples until balanced), oversampling minority (duplicate or SMOTE-synthesize minority examples), class weighting (multiply minority loss by n_majority/n_minority)
- Threshold tuning: move decision threshold from 0.5 to balance precision/recall; always tune on validation set

### Multilabel Pattern
- Problem: each example can belong to multiple classes simultaneously (article about sports AND politics)
- Solution: replace softmax + cross-entropy with independent sigmoid per class + binary cross-entropy per class
- Loss: L = -Σₖ [yₖ log(ŷₖ) + (1-yₖ) log(1-ŷₖ)]; optimizes each label independently
- Evaluation: per-class AUC; micro/macro averaged F1; Hamming loss

### Ensembles Pattern
- Problem: a single model has high variance (random forest) or high bias (linear model); combining models reduces both
- Bagging: train N models on bootstrap samples; average predictions; reduces variance
- Boosting: train sequential models focused on errors of previous; reduces bias
- Stacking: use predictions of N base models as features for a meta-learner; learns which base model to trust
- When to use: accuracy is critical, compute budget allows training multiple models, interpretability is not required

### Cascade Pattern
- Problem: one complex model tries to handle all cases, including easy ones that don't need it; expensive and slow
- Solution: chain simpler models in sequence; first model handles easy cases and routes hard cases to next model
- Example: spam filter → rule-based for obvious spam → simple classifier for moderate cases → complex model for hard cases
- Trade-off: adds system complexity; each stage's errors propagate; latency depends on fraction reaching each stage

### Neutral Class Pattern
- Problem: forcing binary classification on inherently ambiguous examples (borderline sentiment, uncertain medical diagnosis) produces noisy training signal and incorrect confident predictions
- Solution: add a "neutral" or "uncertain" class; train model to predict neutral when evidence is insufficient
- Effect: model learns to be uncertain in uncertain cases; reduces calibration error; improves actionability

**What the book gets right / what to watch out for**
The neutral class pattern is underused and practically important — forcing a binary prediction on ambiguous cases trains the model to be confidently wrong. The cascade pattern correctly identifies that most examples in production are "easy" and don't require an expensive model — routing reduces compute cost and latency. Boosting (XGBoost/LightGBM) is the correct default for tabular data regardless of class balance.

---

## Chapter 4: Model Training Patterns

**The problem the book is addressing**
Training bugs are silent — wrong learning rate, missing regularization, incorrect loss function — the model trains without errors but converges to a suboptimal solution. Practitioners need a checklist of training decisions and the patterns that handle each correctly.

**The core insight**
The training loop has four components: initialization, forward pass, loss computation, and parameter update. Each has multiple design choices with known trade-offs. Custom loss functions can encode domain knowledge (asymmetric cost of false positives vs false negatives) that standard losses ignore.

**The mechanics**
- Training loop: `for batch in dataloader: optimizer.zero_grad(); output = model(batch.x); loss = criterion(output, batch.y); loss.backward(); optimizer.step()`
- Batching: mini-batch (B=32–256); larger batches → smoother gradients but lower generalization; gradient accumulation when GPU memory limits batch size
- Early stopping: monitor val loss; stop when it doesn't improve for N epochs; `restore_best_weights=True`
- LR scheduling: cosine annealing (smooth), step decay, warmup (linear increase for first 2–5% of training)
- Custom loss functions: fraud detection — penalize false negatives more than false positives; `loss = pos_weight * y * log(ŷ) + (1-y) * log(1-ŷ)` with pos_weight > 1
- Transfer learning: freeze pretrained layers → train new layers → unfreeze all → fine-tune at lower LR
- Regularization: L1 (sparsity), L2 (small weights), dropout (random deactivation), data augmentation

**What the book gets right / what to watch out for**
Asymmetric loss functions are a simple and powerful way to encode business requirements — the cost of missing a fraud transaction is not equal to the cost of a false alarm, and the loss function should reflect this. Weight decay (L2 regularization) should be applied to weights but not to biases or normalization layer parameters — most frameworks handle this correctly with AdamW.

---

## Chapter 5: Deployment Patterns

**The problem the book is addressing**
Model deployment is not a single event — it is a continuous process of updating models, testing changes safely, and responding to failures. Deployment without systematic safety mechanisms causes silent degradation or outright failures that affect all users simultaneously.

**The core insight**
Safe deployment separates validation from rollout. Shadow mode tests a new model without exposing predictions to users. Blue-green deployments maintain a known-good version while testing a new one. Canary deployments gradually increase traffic to the new version with the ability to roll back instantly.

**The mechanics**

### Keyed Predictions Pattern
- Problem: batch inference produces predictions without traceability back to the source record
- Solution: pass a unique key (row ID, user ID, request ID) through the prediction pipeline; output (key, prediction) pairs
- Enables: joining predictions back to source data for analysis, debugging specific predictions, deduplication in streaming pipelines

### Continuous Deployment — Kubeflow/TFX
- Kubeflow Pipelines: DAG of ML steps (data validation, preprocessing, training, evaluation, deployment); each step is a containerized function; automatic dependency management
- TFX: TensorFlow Extended; similar concept with TFX components; ExampleGen → StatisticsGen → SchemaGen → ExampleValidator → Transform → Trainer → Evaluator → Pusher
- Model validation gate: only deploy if new model exceeds current production model on evaluation metrics

### Blue-Green Deployment
- Maintain two identical production environments (blue = current, green = new model)
- Route all traffic to blue; deploy and test green; switch DNS/load balancer to green; blue becomes standby
- Instant rollback: switch load balancer back to blue if issues detected
- Cost: running two identical environments simultaneously doubles infrastructure cost during rollout

### Shadow Mode Testing
- New model receives all production requests; runs inference; logs predictions; predictions never shown to users
- Compare offline: are shadow model predictions better than production model predictions?
- Safe: no user impact; captures real production traffic distribution
- Limitation: doesn't measure impact on business metrics (no user interaction with shadow predictions)

### Managing Retraining and Model Drift
- Scheduled retraining: retrain on fixed schedule (weekly, monthly); simple but may miss rapid drift
- Metric-triggered retraining: monitor model performance metrics; trigger retraining when metric drops below threshold
- Data-triggered retraining: monitor feature distributions; trigger retraining when statistical test detects drift
- Monitoring: log prediction distribution, feature distributions, business metrics; alert on anomalies

**What the book gets right / what to watch out for**
Shadow mode is the safest deployment pattern and should be the default for high-stakes model updates. The keyed predictions pattern is essential for auditability — regulated industries (finance, healthcare) require traceability from prediction to input. Blue-green deployment is expensive but provides instant rollback — use it for high-traffic, low-risk-tolerance services. Canary deployment is more efficient for most cases.

---

## Chapter 6: Repeatability Patterns

**The problem the book is addressing**
An ML experiment that can't be reproduced is not science — it's a lucky result. Training-serving skew (different preprocessing at training vs serving time) silently degrades model performance. Without systematic reproducibility, debugging production issues is nearly impossible.

**The core insight**
Repeatability requires that: the same code produces the same result (version control + deterministic seeds), the same data produces the same result (data versioning), and training and serving apply identical preprocessing (transform pattern). Training-serving skew is the most common and most silent ML production bug.

**The mechanics**

### Transform Pattern
- Problem: feature preprocessing computed at training time (scaling parameters, vocabulary mappings) must be applied identically at serving time; differences cause training-serving skew
- Solution: export the preprocessing transformation as part of the model artifact; TFX `Transform` component exports a preprocessing function that is applied identically at train and serve
- scikit-learn equivalent: always save the entire Pipeline (preprocessor + model), not just the model
- Common skew bugs: computing mean/std on full dataset (including val/test) instead of train only; different tokenization at train vs serve; different categorical encoding mappings

### Reproducibility
- Fixed random seeds: `torch.manual_seed(42)`, `np.random.seed(42)`, `random.seed(42)` — control all sources of randomness
- Version control: code in git; data with DVC or MLflow; environments with Docker
- Containerization: Docker image captures exact library versions; eliminates "works on my machine" problems
- Experiment logging: log seed, hyperparameters, data version, code commit hash for every run

### Handling Data Drift
- Monitor feature distributions: compare mean, std, quantiles of each feature between reference window and current window
- Statistical tests: KS test (continuous features), chi-squared (categorical features); PSI > 0.25 = significant shift
- Retraining strategy: retrain on data that includes the shifted distribution; expand training window vs sliding window trade-off

### Batch vs Online Predictions
- Batch: run model on entire dataset at once; store predictions in database; serve from cache; low latency, potentially stale
- Online: run model on each request in real time; always fresh predictions; higher latency, higher compute cost
- Hybrid: pre-compute batch predictions for high-frequency entities; use online for new or low-frequency entities

### Orchestrated Pipelines
- Airflow: general-purpose DAG scheduler; each DAG is a Python file; operators for Spark, BigQuery, Docker, etc.
- Automation: once pipeline is defined as a DAG, it can be triggered on schedule, on data arrival, or on metric threshold
- Reproducibility: a pipeline run captures inputs, outputs, and timing for each step — enables debugging and auditing

**What the book gets right / what to watch out for**
Training-serving skew is the single most impactful production ML bug — it is silent, pervasive, and often discovered only months after deployment when model performance is already degraded. The transform pattern (export preprocessing with the model) is the correct prevention. Containerization + version control are the minimum bar for reproducibility; adding data versioning (DVC, Delta Lake) is required for serious production systems.

## Flashcards

**Data quality dimensions?** #flashcard
accuracy (correct values), completeness (no missing fields), consistency (same format across systems), timeliness (data available when needed)

**ML lifecycle phases?** #flashcard
discovery (problem definition, data exploration) → development (feature engineering, training, evaluation) → deployment (serving, monitoring, maintenance)

**AI readiness levels?** #flashcard
tactical (point solutions, individual models), strategic (platform, shared infrastructure), transformational (ML as core business capability)

**Reproducibility?** #flashcard
same code + same data + same environment → same result; requires version control for code, data, and dependencies

**Standardization?** #flashcard
(x - μ) / σ → zero mean unit variance; required for distance-based and gradient-sensitive models

**Min-max normalization?** #flashcard
(x - x_min) / (x_max - x_min) → [0,1]; use when bounds are known

**Log transform?** #flashcard
log(1+x) for right-skewed features; brings outliers closer to the bulk of the distribution

**Problem: high-cardinality categoricals (user IDs, product SKUs) can't be one-hot encoded?** #flashcard
vocabulary too large, cold-start for new values

**Solution?** #flashcard
hash(feature_value) % num_buckets maps any value to a fixed number of buckets; works for new values at inference

**Trade-off?** #flashcard
bucket collisions (two different values map to the same bucket) introduce noise; larger bucket count reduces collisions but increases parameters

**Use when?** #flashcard
feature cardinality > 10,000, or new values appear at serving time (cold-start problem)

**Problem?** #flashcard
a continuous output (regression) can be reframed as a classification over buckets; a binary label can be reframed as a regression over probability

**Solution?** #flashcard
regression → classification: bucket the output into ranges (e.g., rainfall in mm → {no rain, light, moderate, heavy}); lose ranking within buckets but gain more informative loss signal

**When to reframe?** #flashcard
when the distribution of the target is multimodal, or when you need calibrated interval predictions rather than point predictions

**Problem?** #flashcard
linear models can't capture interactions between features (user_age × product_category effects)

**Solution?** #flashcard
explicitly create a new feature that is the Cartesian product of two categorical features: hour_of_day × day_of_week → 168 combinations

**Trade-off?** #flashcard
multiplies feature space size; increases risk of overfitting; most useful for linear models

**Modern alternative?** #flashcard
embedding-based models (neural networks) learn feature interactions implicitly

**Problem?** #flashcard
real-world entities are described by multiple data types (product: image + text description + numerical price)

**Solution?** #flashcard
separate encoders per modality (CNN for image, BERT for text, FC for numerical); concatenate latent representations; joint fine-tuning

**Late fusion vs early fusion?** #flashcard
late fusion (concatenate after encoding) is simpler and handles missing modalities; early fusion (combine raw features) requires all modalities

**Problem?** #flashcard
class imbalance causes models to optimize for majority class; minority class performance is poor

**Solutions?** #flashcard
downsampling majority (discard majority examples until balanced), oversampling minority (duplicate or SMOTE-synthesize minority examples), class weighting (multiply minority loss by n_majority/n_minority)

**Threshold tuning?** #flashcard
move decision threshold from 0.5 to balance precision/recall; always tune on validation set

**Problem?** #flashcard
each example can belong to multiple classes simultaneously (article about sports AND politics)

**Solution?** #flashcard
replace softmax + cross-entropy with independent sigmoid per class + binary cross-entropy per class

**Loss?** #flashcard
L = -Σₖ [yₖ log(ŷₖ) + (1-yₖ) log(1-ŷₖ)]; optimizes each label independently

**Evaluation?** #flashcard
per-class AUC; micro/macro averaged F1; Hamming loss

**Problem?** #flashcard
a single model has high variance (random forest) or high bias (linear model); combining models reduces both

**Bagging?** #flashcard
train N models on bootstrap samples; average predictions; reduces variance

**Boosting?** #flashcard
train sequential models focused on errors of previous; reduces bias

**Stacking?** #flashcard
use predictions of N base models as features for a meta-learner; learns which base model to trust

**When to use?** #flashcard
accuracy is critical, compute budget allows training multiple models, interpretability is not required

**Problem?** #flashcard
one complex model tries to handle all cases, including easy ones that don't need it; expensive and slow

**Solution?** #flashcard
chain simpler models in sequence; first model handles easy cases and routes hard cases to next model

**Example?** #flashcard
spam filter → rule-based for obvious spam → simple classifier for moderate cases → complex model for hard cases

**Trade-off?** #flashcard
adds system complexity; each stage's errors propagate; latency depends on fraction reaching each stage

**Problem?** #flashcard
forcing binary classification on inherently ambiguous examples (borderline sentiment, uncertain medical diagnosis) produces noisy training signal and incorrect confident predictions

**Solution?** #flashcard
add a "neutral" or "uncertain" class; train model to predict neutral when evidence is insufficient

**Effect?** #flashcard
model learns to be uncertain in uncertain cases; reduces calibration error; improves actionability

**Training loop?** #flashcard
for batch in dataloader: optimizer.zero_grad(); output = model(batch.x); loss = criterion(output, batch.y); loss.backward(); optimizer.step()

**Batching?** #flashcard
mini-batch (B=32–256); larger batches → smoother gradients but lower generalization; gradient accumulation when GPU memory limits batch size

**Early stopping?** #flashcard
monitor val loss; stop when it doesn't improve for N epochs; restore_best_weights=True

**LR scheduling?** #flashcard
cosine annealing (smooth), step decay, warmup (linear increase for first 2–5% of training)

**Custom loss functions: fraud detection?** #flashcard
penalize false negatives more than false positives; loss = pos_weight  y  log(ŷ) + (1-y) * log(1-ŷ) with pos_weight > 1

**Transfer learning?** #flashcard
freeze pretrained layers → train new layers → unfreeze all → fine-tune at lower LR

**Regularization?** #flashcard
L1 (sparsity), L2 (small weights), dropout (random deactivation), data augmentation

**Problem?** #flashcard
batch inference produces predictions without traceability back to the source record

**Solution?** #flashcard
pass a unique key (row ID, user ID, request ID) through the prediction pipeline; output (key, prediction) pairs

**Enables?** #flashcard
joining predictions back to source data for analysis, debugging specific predictions, deduplication in streaming pipelines

**Kubeflow Pipelines?** #flashcard
DAG of ML steps (data validation, preprocessing, training, evaluation, deployment); each step is a containerized function; automatic dependency management

**TFX?** #flashcard
TensorFlow Extended; similar concept with TFX components; ExampleGen → StatisticsGen → SchemaGen → ExampleValidator → Transform → Trainer → Evaluator → Pusher

**Model validation gate?** #flashcard
only deploy if new model exceeds current production model on evaluation metrics

**Maintain two identical production environments (blue = current, green = new model)?** #flashcard
Maintain two identical production environments (blue = current, green = new model)

**Route all traffic to blue; deploy and test green; switch DNS/load balancer to green; blue becomes standby?** #flashcard
Route all traffic to blue; deploy and test green; switch DNS/load balancer to green; blue becomes standby

**Instant rollback?** #flashcard
switch load balancer back to blue if issues detected

**Cost?** #flashcard
running two identical environments simultaneously doubles infrastructure cost during rollout

**New model receives all production requests; runs inference; logs predictions; predictions never shown to users?** #flashcard
New model receives all production requests; runs inference; logs predictions; predictions never shown to users

**Compare offline?** #flashcard
are shadow model predictions better than production model predictions?

**Safe?** #flashcard
no user impact; captures real production traffic distribution

**Limitation?** #flashcard
doesn't measure impact on business metrics (no user interaction with shadow predictions)

**Scheduled retraining?** #flashcard
retrain on fixed schedule (weekly, monthly); simple but may miss rapid drift

**Metric-triggered retraining?** #flashcard
monitor model performance metrics; trigger retraining when metric drops below threshold

**Data-triggered retraining?** #flashcard
monitor feature distributions; trigger retraining when statistical test detects drift

**Monitoring?** #flashcard
log prediction distribution, feature distributions, business metrics; alert on anomalies

**Problem?** #flashcard
feature preprocessing computed at training time (scaling parameters, vocabulary mappings) must be applied identically at serving time; differences cause training-serving skew

**Solution?** #flashcard
export the preprocessing transformation as part of the model artifact; TFX Transform component exports a preprocessing function that is applied identically at train and serve

**scikit-learn equivalent?** #flashcard
always save the entire Pipeline (preprocessor + model), not just the model

**Common skew bugs?** #flashcard
computing mean/std on full dataset (including val/test) instead of train only; different tokenization at train vs serve; different categorical encoding mappings

**Fixed random seeds: torch.manual_seed(42), np.random.seed(42), random.seed(42)?** #flashcard
control all sources of randomness

**Version control?** #flashcard
code in git; data with DVC or MLflow; environments with Docker

**Containerization?** #flashcard
Docker image captures exact library versions; eliminates "works on my machine" problems

**Experiment logging?** #flashcard
log seed, hyperparameters, data version, code commit hash for every run

**Monitor feature distributions?** #flashcard
compare mean, std, quantiles of each feature between reference window and current window

**Statistical tests?** #flashcard
KS test (continuous features), chi-squared (categorical features); PSI > 0.25 = significant shift

**Retraining strategy?** #flashcard
retrain on data that includes the shifted distribution; expand training window vs sliding window trade-off

**Batch?** #flashcard
run model on entire dataset at once; store predictions in database; serve from cache; low latency, potentially stale

**Online?** #flashcard
run model on each request in real time; always fresh predictions; higher latency, higher compute cost

**Hybrid?** #flashcard
pre-compute batch predictions for high-frequency entities; use online for new or low-frequency entities

**Airflow?** #flashcard
general-purpose DAG scheduler; each DAG is a Python file; operators for Spark, BigQuery, Docker, etc.

**Automation?** #flashcard
once pipeline is defined as a DAG, it can be triggered on schedule, on data arrival, or on metric threshold

**Reproducibility: a pipeline run captures inputs, outputs, and timing for each step?** #flashcard
enables debugging and auditing
