---
module: References
topic: Book Notes
subtopic: Mlops Machine Learning Engineering
status: unread
tags: [references, ml, book-notes-mlops]
---
# Machine Learning Engineering

## Chapter 1: Introduction — What is ML Engineering

**The problem the book is addressing**
Data scientists build models; ML engineers ship them. The gap between a working notebook model and a production system that serves millions of predictions reliably is vast and poorly documented. Most ML education stops at model training.

**The core insight**
ML engineering is the discipline of building, deploying, and maintaining ML systems in production. It combines software engineering (APIs, testing, version control, monitoring) with ML-specific concerns (data validation, model evaluation, training pipelines, serving infrastructure). An ML engineer is responsible for the entire lifecycle, not just the model.

**The mechanics**
- Mathematical notation: scalar (x), vector (x), matrix (X), set (S), Euclidean norm ||x||₂ = √(Σxᵢ²)
- ML definition: a program is said to learn from experience E with respect to task T and performance measure P if P improves with E (Mitchell's definition)
- Learning paradigms: supervised (labeled), unsupervised (structure), semi-supervised (few labels + many unlabeled), RL (reward signal from environment)
- Model families: model-based (parameterized function, fast inference), instance-based (kNN, stores training data), deep vs shallow
- Project lifecycle: goal definition → data collection → feature engineering → model training → evaluation → deployment → serving → monitoring → maintenance

**What the book gets right / what to watch out for**
The lifecycle framing — treating ML as an engineering discipline with a full system lifecycle — is the book's core contribution. Most data science education covers only training and evaluation; this book covers the full system. The notation section is important for reading papers — practitioners who skip mathematical notation can't evaluate algorithm claims independently.

---

## Chapter 2: Before the Project Starts

**The problem the book is addressing**
Most ML projects fail not because of bad models but because the problem was poorly defined, the team was misaligned, or the complexity was underestimated. Fixing these issues after months of work is expensive.

**The core insight**
Prioritize ML projects by expected impact vs estimated cost. Progress in ML is non-linear — the first 80% of performance is achievable quickly; the last 20% often requires 10× the effort. Goal definition and team structure decisions made before the first line of code determines whether the project succeeds.

**The mechanics**
- Prioritization: `priority = expected_impact / estimated_cost`; favor high-impact, low-complexity projects for first ML deployments
- Complexity estimation: non-linear progress — simple approaches achieve 80% of maximum performance quickly; avoiding the last 20% is usually the right business decision
- Goal definition: what specific metric will improve? What is the baseline? What is the minimum acceptable performance for production?
- Team structure: collaborative (ML and product teams work closely), integrated (ML engineers embedded in product teams) — integrated produces better alignment
- Common project failures: lack of ML experience (underestimate complexity), lack of data infrastructure (data not accessible or queryable), misaligned teams (ML builds what product doesn't want)

**What the book gets right / what to watch out for**
The non-linear complexity curve is one of the most important planning insights — teams that commit to 99% accuracy when 90% would deliver the business value waste resources on the hardest part of the problem. The team structure argument for embedded ML engineers over siloed data science teams reflects industry experience at successful ML companies (Google, Netflix).

---

## Chapter 3: Data Collection and Preparation

**The problem the book is addressing**
"Garbage in, garbage out" is well-known but systematically ignored. Data quality problems — bias, low predictive power, leakage, outdated labels — produce models that underperform in production without clear error signals during development.

**The core insight**
Before modeling, answer five questions about your data: Is it accessible (can you actually read it)? Is it sizeable (enough examples of each class)? Is it usable (correct labels, no massive noise)? Is it understandable (you know what each feature means)? Is it reliable (labels were created consistently)? Failing any of these disqualifies the dataset from production use.

**The mechanics**
- Five data questions: accessible, sizeable (≥1000 examples per class for classification), usable, understandable, reliable
- Common data problems: collection cost (expensive to label, slow to collect), selection bias (training data doesn't represent deployment population), low predictive power (features don't correlate with target), outdated labels (world changed since labeling), data leakage (future information in features)
- Good data properties: collected without bias, representative of the deployment population, labeled consistently (inter-annotator agreement > 0.8 Cohen's kappa)
- Data partitioning: training (60%), validation (20%), test (20%); test set is never used for any decisions
- Imputation: mean/median for continuous, mode for categorical; consider model-based imputation (IterativeImputer) for high-value features
- Class imbalance: oversampling (SMOTE, duplicate minority), undersampling (discard majority), cost-sensitive learning
- Sampling strategies: simple random, systematic (every Kth element), stratified (preserves class ratios), cluster (sample whole groups)
- Data versioning: DVC, Delta Lake, or timestamped snapshots — know exactly which data produced which model

**What the book gets right / what to watch out for**
"Data first, algorithm second" is the most important principle in the book. The five-question framework is a practical checklist — teams should answer all five before committing to a modeling approach. Data versioning is not optional for production systems — you need to reproduce any model from six months ago to debug production issues.

---

## Chapter 4: Feature Engineering

**The problem the book is addressing**
Raw features rarely contain the signal in the form the model needs. Manual feature engineering is the primary way practitioners inject domain knowledge. The same data with good features and a simple model often outperforms a complex model on poor features.

**The core insight**
Good features have four properties: high predictive power (correlated with target), fast to compute (available at serving time within latency budget), reliable (consistent across data sources and time), and not too correlated with other features (redundancy wastes capacity). Feature selection is as important as feature construction.

**The mechanics**
- Text features: one-hot encoding (binary bag-of-words), TF-IDF (frequency weighted), word embeddings (dense semantic vectors)
- Feature hashing trick: map arbitrary feature values to fixed-size vector; `hash(feature_value) % D`; handles OOV without vocabulary
- Feature stacking: concatenate features from different sources into a single vector
- Feature selection: remove features in the long tail (low frequency, low predictive power); use permutation importance or SHAP to rank
- Feature discretization: bin continuous feature into categories; uniform binning (equal-width intervals), k-means binning (cluster-based), quantile binning (equal-frequency)
- Word embeddings: Word2Vec skip-gram, GloVe, fastText; pre-trained vectors encode semantic similarity; use as feature initialization
- Data leakage prevention: never use information from validation/test time to construct training features; be especially careful with time-based features
- Feature documentation: for each feature, document: definition, computation method, expected range, known correlations, production availability

**What the book gets right / what to watch out for**
Feature documentation is the most overlooked engineering practice — teams discover undocumented features in production that have silent dependencies on other systems. Feature reliability is often more important than predictive power — a feature that is 10% better on average but occasionally unavailable at serving time is worse than a reliable 5% better feature.

---

## Chapter 5: Supervised Model Training

**The problem the book is addressing**
Jumping to complex models before understanding achievable performance and establishing baselines wastes time. Without a baseline, you don't know if your complex model is actually adding value.

**The core insight**
Training proceeds in three phases: establish achievable performance (what's the best a human could do?), establish a baseline (what does the simplest possible model do?), then improve from there. Algorithm selection should be systematic — start with gradient boosted trees for tabular data, then consider neural networks only if they demonstrably improve over the baseline.

**The mechanics**
- Schema validation: before training, validate that features have expected types, ranges, and distributions; fail loudly on violations
- Achievable performance: human accuracy on the task; defines the ceiling
- Performance metric selection: precision/recall/F1 for classification; RMSE/MAE for regression; choose the metric that aligns with business cost
- Baseline: random classifier (majority class), mean predictor (regression), linear model; must be beaten to justify any complexity
- Label representation: multiclass → one-hot encoding; multilabel → bag-of-words binary vector
- Algorithm selection: tabular data → gradient boosting (XGBoost/LightGBM); images → CNN or ViT; text → BERT/GPT fine-tune; time series → LightGBM with lag features or N-BEATS
- Hyperparameter tuning: random search with log-uniform distributions is the efficient approach; Optuna/Hyperopt for Bayesian optimization
- Bias-variance tradeoff: underfitting (high bias) → increase model capacity or train longer; overfitting (high variance) → more data, regularization, or reduce capacity

**What the book gets right / what to watch out for**
The baseline-first discipline separates engineering from tinkering. Teams that build baselines before complex models discover in 20% of cases that the baseline is good enough, saving months of work. The algorithm selection guidance is practical and correct for current tooling — LightGBM is the correct default for tabular data, not neural networks.

---

## Chapter 6: Neural Network Training

**The problem the book is addressing**
Neural network training has more degrees of freedom than classical ML — architecture, initialization, optimizer, batch size, regularization — and failures are silent. A misconfigured training run produces a model that is worse than a simple baseline with no clear error.

**The core insight**
Neural network training requires explicit decisions about five components: objective function (must match the task), initialization (Xavier/He — determines whether gradients flow), optimizer (Adam for most tasks, SGD+momentum for vision), regularization (dropout + weight decay + data augmentation), and batch size (larger is not always better — small batches provide regularization through gradient noise).

**The mechanics**
- Performance metric vs cost function: accuracy is not differentiable; cross-entropy loss approximates it and is differentiable
- Cost functions: MSE for regression, cross-entropy for classification, Dice loss for segmentation
- Weight initialization: Xavier (Var[w] = 2/(fan_in+fan_out)) for tanh/sigmoid; He (Var[w] = 2/fan_in) for ReLU — prevents vanishing/exploding gradients at initialization
- Optimization: Adam (β₁=0.9, β₂=0.999, ε=1e-8) default; AdamW (decoupled weight decay) preferred; SGD+momentum for vision tasks
- Hyperparameter tuning: LR is the most important; try {1e-1, 1e-2, 1e-3, 1e-4} in log space; use LR warmup + cosine decay
- Transfer learning: load pretrained weights; freeze backbone; train head; unfreeze; fine-tune at lower LR (10-100× smaller)
- Regularization: L2 weight decay (λ=1e-4 typical), dropout (p=0.5 for FC, p=0.1-0.2 for conv), batch normalization (implicit regularization), data augmentation
- Batch size: B=256 is a reasonable default; larger batches → lower variance gradients but weaker regularization and worse generalization; scale LR linearly with batch size when scaling

**What the book gets right / what to watch out for**
Weight initialization is often overlooked but critical — wrong initialization causes training to fail from the start with no error message. He initialization for ReLU networks is well-established; using the wrong initialization (e.g., Xavier for ReLU) causes vanishing activations. The LR-batch size linear scaling rule (Goyal et al.) is correct for large-batch training but breaks down for very large batches.

---

## Chapter 7: Model Evaluation

**The problem the book is addressing**
A model with good offline metrics can fail catastrophically in production. Offline evaluation must approximate production conditions as closely as possible, and online evaluation must measure the metrics that actually matter to the business.

**The core insight**
Offline evaluation is necessary but not sufficient. Online evaluation (A/B testing, bandits) measures actual business impact. The gap between offline and online metrics reveals systematic biases in the evaluation setup. Multi-armed bandits are strictly better than A/B tests when the goal is to maximize cumulative performance rather than just determine which variant is better.

**The mechanics**
- Offline evaluation: confusion matrix (classification), RMSE/MAE (regression), AUC-ROC (ranking), BLEU/ROUGE (generation)
- Validation vs test: validation for all iterative model decisions; test set used exactly once for final evaluation
- A/B testing: split users into treatment/control; measure primary metric + guardrail metrics; need statistical significance (p < 0.05) with adequate power (β > 0.8)
- Multi-armed bandit: Thompson sampling — maintain a Beta distribution per arm; sample from each distribution; choose arm with highest sample; update distribution with observed outcome
- Exploration vs exploitation: pure exploitation = A/B test (miss out during test period); bandit allocates more traffic to better arm as evidence accumulates
- Online evaluation business metrics: revenue, engagement, retention — these are the ultimate measures of model value

**What the book gets right / what to watch out for**
The multi-armed bandit framing for A/B testing is correct when the goal is to maximize cumulative reward — bandits eliminate the "regret" of running a losing variant for the duration of the test. A/B testing is appropriate when the goal is a clean causal estimate of the effect (academic/regulatory contexts). Both require sufficient traffic — bandits converge faster but still need time.

---

## Chapter 8: Deployment — Static, Dynamic, and Canary

**The problem the book is addressing**
Deployment means the model runs in a live system serving real users. The deployment strategy determines how safely you can update models, what latency guarantees you can make, and what happens when the model fails.

**The core insight**
Static deployment (model compiled into application code) is fast and private but slow to update. Dynamic deployment (model served via API) enables rapid updates and version management. Canary deployment is the safe update mechanism — gradually route traffic to the new model while monitoring for regressions.

**The mechanics**
- Static deployment: serialize model to file (pickle/ONNX/TorchScript); load as DLL/shared library or embed in application; advantages: low latency (no network), no external dependency, works offline; disadvantages: slow to update (requires application redeployment), hard to A/B test
- Dynamic deployment: model server (REST/gRPC API); request comes in, model runs inference, response returned; advantages: rapid updates, A/B testing, monitoring; disadvantages: network latency, infrastructure dependency
- Streaming deployment: process events as they arrive (Kafka consumer + model); low latency, stateful
- Canary deployment: route 1–5% of traffic to new model version; monitor metrics; gradually increase to 100% or rollback
- Multi-armed bandit for deployment: allocate traffic adaptively based on observed performance — converges faster than fixed canary schedule
- Automation: version control model artifacts; automated testing (unit, integration, performance) before deployment; approval gate for production

**What the book gets right / what to watch out for**
The static vs dynamic deployment distinction has important security implications — static deployment keeps model weights on-device and prevents model extraction attacks. Dynamic deployment enables faster iteration but requires infrastructure uptime. Canary deployment is the correct default update mechanism — never push directly to 100% traffic for ML model updates.

---

## Chapter 9: Model Serving, Monitoring, and Maintenance

**The problem the book is addressing**
A deployed model is not done. It must be monitored continuously, maintained as the world changes, and defended against adversarial inputs. These operational concerns are outside most ML education but consume most of ML engineering time in production.

**The core insight**
Model serving quality is measured by more than accuracy: security (can the model be fooled or extracted?), ease of deployment (how long does it take to update?), validity guarantees (what input types does it handle?), rollback capability (can you revert quickly?), and absence of training-serving skew. All six must be considered when designing serving infrastructure.

**The mechanics**
- Serving properties: security (input validation, model encryption), ease of deployment (CI/CD pipeline), validity guarantees (input schema validation), rollback (version registry, instant rollback), no training-serving skew (same preprocessing code)
- Hidden feedback loops: model predictions influence future data → future training data → next model; untreated, leads to self-reinforcing biases
- Batch vs on-demand: batch (nightly job, cached results) for throughput; on-demand (per-request inference) for freshness
- Performance drift: monitor prediction distribution and business metrics weekly; alert on statistical deviation
- Adversarial inputs: users may craft inputs to fool the model (adversarial examples, prompt injection); input validation and anomaly detection at the API layer
- Model maintenance: retrain when drift detected, when new data labels become available, or on a regular schedule; keep retraining pipeline automated and reproducible

**What the book gets right / what to watch out for**
Hidden feedback loops are the most dangerous long-term failure mode in production ML — the model shapes the world it's trained on. Content recommendation models that promote engaging content eventually degrade toward extreme content because it maximizes engagement metrics. Detecting and breaking these loops requires measuring leading indicators of loop formation, not just model metrics.

---

## Chapter 10: Conclusion — ML as Mainstream Engineering

**The problem the book is addressing**
ML practitioners treat their discipline as special or separate from software engineering. This leads to poor engineering practices (no version control, no testing, no monitoring) that are standard in software but missing from ML workflows.

**The core insight**
ML is becoming a mainstream software engineering discipline. The same principles that make software reliable — version control, testing, documentation, monitoring, automation — apply to ML systems. The most common failures are not modeling failures but engineering failures: data issues, technical debt, and lack of understanding of what the model is actually doing.

**The mechanics**
- Common pitfalls: data quality issues (garbage in, garbage out), technical debt (hardcoded paths, undocumented features, untested preprocessing), lack of understanding (shipping a model you can't explain)
- Engineering standards for ML: version control (code + data + models), unit tests for preprocessing and postprocessing, integration tests for serving, monitoring dashboards, runbooks for common incidents
- Democratization: open-source frameworks (PyTorch, sklearn, HuggingFace), cloud ML platforms (SageMaker, Vertex AI), AutoML tools lower the barrier to entry
- Full lifecycle ownership: ML engineers own the model from conception to retirement; this includes monitoring, retraining, and deprecation

**What the book gets right / what to watch out for**
The engineering discipline framing is the most important cultural argument in the book. Teams that treat ML as research (no tests, no monitoring, no reproducibility) create systems they can't maintain. Democratization lowers the bar to deploying models but not the bar to deploying them responsibly — AutoML makes it easy to ship a model that is biased, uncalibrated, or silently degrading.

## Flashcards

**Mathematical notation?** #flashcard
scalar (x), vector (x), matrix (X), set (S), Euclidean norm ||x||₂ = √(Σxᵢ²)

**ML definition?** #flashcard
a program is said to learn from experience E with respect to task T and performance measure P if P improves with E (Mitchell's definition)

**Learning paradigms?** #flashcard
supervised (labeled), unsupervised (structure), semi-supervised (few labels + many unlabeled), RL (reward signal from environment)

**Model families?** #flashcard
model-based (parameterized function, fast inference), instance-based (kNN, stores training data), deep vs shallow

**Project lifecycle?** #flashcard
goal definition → data collection → feature engineering → model training → evaluation → deployment → serving → monitoring → maintenance

**Prioritization?** #flashcard
priority = expected_impact / estimated_cost; favor high-impact, low-complexity projects for first ML deployments

**Complexity estimation: non-linear progress?** #flashcard
simple approaches achieve 80% of maximum performance quickly; avoiding the last 20% is usually the right business decision

**Goal definition?** #flashcard
what specific metric will improve? What is the baseline? What is the minimum acceptable performance for production?

**Team structure: collaborative (ML and product teams work closely), integrated (ML engineers embedded in product teams)?** #flashcard
integrated produces better alignment

**Common project failures?** #flashcard
lack of ML experience (underestimate complexity), lack of data infrastructure (data not accessible or queryable), misaligned teams (ML builds what product doesn't want)

**Five data questions?** #flashcard
accessible, sizeable (≥1000 examples per class for classification), usable, understandable, reliable

**Common data problems?** #flashcard
collection cost (expensive to label, slow to collect), selection bias (training data doesn't represent deployment population), low predictive power (features don't correlate with target), outdated labels (world changed since labeling), data leakage (future information in features)

**Good data properties?** #flashcard
collected without bias, representative of the deployment population, labeled consistently (inter-annotator agreement > 0.8 Cohen's kappa)

**Data partitioning?** #flashcard
training (60%), validation (20%), test (20%); test set is never used for any decisions

**Imputation?** #flashcard
mean/median for continuous, mode for categorical; consider model-based imputation (IterativeImputer) for high-value features

**Class imbalance?** #flashcard
oversampling (SMOTE, duplicate minority), undersampling (discard majority), cost-sensitive learning

**Sampling strategies?** #flashcard
simple random, systematic (every Kth element), stratified (preserves class ratios), cluster (sample whole groups)

**Data versioning: DVC, Delta Lake, or timestamped snapshots?** #flashcard
know exactly which data produced which model

**Text features?** #flashcard
one-hot encoding (binary bag-of-words), TF-IDF (frequency weighted), word embeddings (dense semantic vectors)

**Feature hashing trick?** #flashcard
map arbitrary feature values to fixed-size vector; hash(feature_value) % D; handles OOV without vocabulary

**Feature stacking?** #flashcard
concatenate features from different sources into a single vector

**Feature selection?** #flashcard
remove features in the long tail (low frequency, low predictive power); use permutation importance or SHAP to rank

**Feature discretization?** #flashcard
bin continuous feature into categories; uniform binning (equal-width intervals), k-means binning (cluster-based), quantile binning (equal-frequency)

**Word embeddings?** #flashcard
Word2Vec skip-gram, GloVe, fastText; pre-trained vectors encode semantic similarity; use as feature initialization

**Data leakage prevention?** #flashcard
never use information from validation/test time to construct training features; be especially careful with time-based features

**Feature documentation?** #flashcard
for each feature, document: definition, computation method, expected range, known correlations, production availability

**Schema validation?** #flashcard
before training, validate that features have expected types, ranges, and distributions; fail loudly on violations

**Achievable performance?** #flashcard
human accuracy on the task; defines the ceiling

**Performance metric selection?** #flashcard
precision/recall/F1 for classification; RMSE/MAE for regression; choose the metric that aligns with business cost

**Baseline?** #flashcard
random classifier (majority class), mean predictor (regression), linear model; must be beaten to justify any complexity

**Label representation?** #flashcard
multiclass → one-hot encoding; multilabel → bag-of-words binary vector

**Algorithm selection?** #flashcard
tabular data → gradient boosting (XGBoost/LightGBM); images → CNN or ViT; text → BERT/GPT fine-tune; time series → LightGBM with lag features or N-BEATS

**Hyperparameter tuning?** #flashcard
random search with log-uniform distributions is the efficient approach; Optuna/Hyperopt for Bayesian optimization

**Bias-variance tradeoff?** #flashcard
underfitting (high bias) → increase model capacity or train longer; overfitting (high variance) → more data, regularization, or reduce capacity

**Performance metric vs cost function?** #flashcard
accuracy is not differentiable; cross-entropy loss approximates it and is differentiable

**Cost functions?** #flashcard
MSE for regression, cross-entropy for classification, Dice loss for segmentation

**Weight initialization: Xavier (Var[w] = 2/(fan_in+fan_out)) for tanh/sigmoid; He (Var[w] = 2/fan_in) for ReLU?** #flashcard
prevents vanishing/exploding gradients at initialization

**Optimization?** #flashcard
Adam (β₁=0.9, β₂=0.999, ε=1e-8) default; AdamW (decoupled weight decay) preferred; SGD+momentum for vision tasks

**Hyperparameter tuning?** #flashcard
LR is the most important; try {1e-1, 1e-2, 1e-3, 1e-4} in log space; use LR warmup + cosine decay

**Transfer learning?** #flashcard
load pretrained weights; freeze backbone; train head; unfreeze; fine-tune at lower LR (10-100× smaller)

**Regularization?** #flashcard
L2 weight decay (λ=1e-4 typical), dropout (p=0.5 for FC, p=0.1-0.2 for conv), batch normalization (implicit regularization), data augmentation

**Batch size?** #flashcard
B=256 is a reasonable default; larger batches → lower variance gradients but weaker regularization and worse generalization; scale LR linearly with batch size when scaling

**Offline evaluation?** #flashcard
confusion matrix (classification), RMSE/MAE (regression), AUC-ROC (ranking), BLEU/ROUGE (generation)

**Validation vs test?** #flashcard
validation for all iterative model decisions; test set used exactly once for final evaluation

**A/B testing?** #flashcard
split users into treatment/control; measure primary metric + guardrail metrics; need statistical significance (p < 0.05) with adequate power (β > 0.8)

**Multi-armed bandit: Thompson sampling?** #flashcard
maintain a Beta distribution per arm; sample from each distribution; choose arm with highest sample; update distribution with observed outcome

**Exploration vs exploitation?** #flashcard
pure exploitation = A/B test (miss out during test period); bandit allocates more traffic to better arm as evidence accumulates

**Online evaluation business metrics: revenue, engagement, retention?** #flashcard
these are the ultimate measures of model value

**Static deployment?** #flashcard
serialize model to file (pickle/ONNX/TorchScript); load as DLL/shared library or embed in application; advantages: low latency (no network), no external dependency, works offline; disadvantages: slow to update (requires application redeployment), hard to A/B test

**Dynamic deployment?** #flashcard
model server (REST/gRPC API); request comes in, model runs inference, response returned; advantages: rapid updates, A/B testing, monitoring; disadvantages: network latency, infrastructure dependency

**Streaming deployment?** #flashcard
process events as they arrive (Kafka consumer + model); low latency, stateful

**Canary deployment?** #flashcard
route 1–5% of traffic to new model version; monitor metrics; gradually increase to 100% or rollback

**Multi-armed bandit for deployment: allocate traffic adaptively based on observed performance?** #flashcard
converges faster than fixed canary schedule

**Automation?** #flashcard
version control model artifacts; automated testing (unit, integration, performance) before deployment; approval gate for production

**Serving properties?** #flashcard
security (input validation, model encryption), ease of deployment (CI/CD pipeline), validity guarantees (input schema validation), rollback (version registry, instant rollback), no training-serving skew (same preprocessing code)

**Hidden feedback loops?** #flashcard
model predictions influence future data → future training data → next model; untreated, leads to self-reinforcing biases

**Batch vs on-demand?** #flashcard
batch (nightly job, cached results) for throughput; on-demand (per-request inference) for freshness

**Performance drift?** #flashcard
monitor prediction distribution and business metrics weekly; alert on statistical deviation

**Adversarial inputs?** #flashcard
users may craft inputs to fool the model (adversarial examples, prompt injection); input validation and anomaly detection at the API layer

**Model maintenance?** #flashcard
retrain when drift detected, when new data labels become available, or on a regular schedule; keep retraining pipeline automated and reproducible

**Common pitfalls?** #flashcard
data quality issues (garbage in, garbage out), technical debt (hardcoded paths, undocumented features, untested preprocessing), lack of understanding (shipping a model you can't explain)

**Engineering standards for ML?** #flashcard
version control (code + data + models), unit tests for preprocessing and postprocessing, integration tests for serving, monitoring dashboards, runbooks for common incidents

**Democratization?** #flashcard
open-source frameworks (PyTorch, sklearn, HuggingFace), cloud ML platforms (SageMaker, Vertex AI), AutoML tools lower the barrier to entry

**Full lifecycle ownership?** #flashcard
ML engineers own the model from conception to retirement; this includes monitoring, retraining, and deprecation
