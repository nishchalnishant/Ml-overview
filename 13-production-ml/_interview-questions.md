---
module: Production ML
topic: Interview Questions
subtopic: ""
status: unread
tags: [productionml, ml, interview-questions, revision]
---

# Production ML — Interview Questions

**For:** SDE-2 / AI Engineer interviews — calibrated to what's actually asked Round 1 and beyond.
**Difficulty guide:**
- **Easy** → Round 1 basics: definitions, standard MLOps concepts, deployment strategies, and basic feature store mechanics. Know these cold.
- **Medium** → Round 2 depth: applied debugging, architectural tradeoffs (streaming vs batch, Lambda vs Kappa), evaluation gotchas, LLM serving optimizations, and monitoring strategies.

---

## Easy

> Round 1 Production ML fundamentals. Standard definitions, MLOps maturity, and deployment patterns.

### Q: What is MLOps and how does it differ from DevOps?
DevOps manages code and infrastructure; MLOps additionally manages data and models, both of which drift over time. A DevOps pipeline is deterministic. An ML pipeline is not: the same code with different training data produces a different model, and a model's quality degrades silently as the world changes. MLOps adds three concerns: data versioning, experiment tracking (hyperparameters/metrics), and continuous performance monitoring (drift detection).

### Q: Describe the stages of MLOps maturity (Level 0 to Level 2).
Level 0 (manual): training in notebooks, manual handoff of a pickle file — high risk, slow iteration. Level 1 (ML pipeline automation): training is automated as a pipeline triggered by new data, but deployment is manual. Level 2 (full CI/CD): both training and deployment pipelines are automated — code/feature changes trigger retraining, validation, and automated deployment (canary) with the same rigor as software CI/CD.

### Q: What is data versioning and why do you need it alongside code versioning?
Two experiments with identical code but different data snapshots produce different models. Without tracking which data version produced which model, you cannot reproduce a result or diagnose regressions. DVC stores a pointer in git referencing a content-hashed blob in remote storage. Full reproducibility requires pinning data, code, library versions, and seeds together.

### Q: Compare blue-green, canary, and shadow deployment for ML models.
**Blue-green**: traffic switches atomically from current (blue) to new (green) environment once validated. Fast but binary (100% risk). **Canary**: new model gets a small % of live traffic (e.g., 5%), progressively ramped with automated health checks gating each step. Limits blast radius. **Shadow (dark launch)**: new model scores 100% of live traffic in parallel, but predictions are only logged, not returned to users. Zero risk, but provides no signal on user reaction/engagement.

### Q: What's the difference between online and offline (batch) inference?
**Online (synchronous)**: serves one request at a time with a tight latency SLA (e.g., fraud check at checkout). **Offline (batch)**: precomputes predictions for all entities on a schedule (nightly recommendations) and stores results in a lookup table. Batch is cheaper (better GPU utilization, no 24/7 uptime) but predictions are stale. Use online when predictions must react to real-time context.

### Q: What should a model card contain, and why is it required for regulated deployments?
A model card documents: intended use case, out-of-scope uses, training data description, known limitations/biases, and evaluation results broken down by subgroups. It allows auditors and downstream teams to assess appropriateness. For regulated deployments (finance, healthcare), it's a compliance requirement for audit trails.

### Q: What is the Baseline Comparison design pattern and why is it non-negotiable?
Before shipping any model, establish what a simple baseline achieves (majority-class predictor, simple heuristic) and report your sophisticated model's *lift* over that baseline. It catches "97% accurate" models on a 97%-majority-class dataset and prevents overengineering if a simple rule captures 90% of the value.

### Q: What are the four core components of a feature store?
(1) **Offline Store**: historical features for training (columnar). (2) **Online Store**: low-latency key-value storage (Redis) for serving current values. (3) **Feature Registry**: catalog of feature definitions with metadata. (4) **Transformation Layer**: single-source-of-truth logic compiled to both batch and streaming paths.

### Q: What is a model registry, and what's its minimum lifecycle state machine?
A centralized catalog of trained model artifacts with versioning and metadata. The minimum state machine is `None → Staging → Production → Archived`. Promoting a new version to Production should automatically archive (not delete) the prior version, preserving it as a rollback target.

### Q: What is ML pipeline orchestration (Airflow vs Kubeflow vs Managed)?
An orchestrator schedules a DAG of steps (data validation, training, deployment). **Airflow** is general-purpose workflow orchestration (ubiquitous but weaker ML-native support). **Kubeflow** is K8s-native and ML-focused (GPU scheduling). **SageMaker/Vertex Pipelines** are fully managed, reducing operational burden but introducing vendor lock-in.

### Q: How do you detect data drift vs concept drift?
**Data drift (covariate shift)**: input feature distribution P(X) changes (e.g., users skew younger) while the X-Y relationship is unchanged. Detect via PSI or KS-test batch-over-batch. **Concept drift**: P(Y|X) changes (e.g., fraud patterns evolve). Detect via live prediction accuracy on delayed labels or streaming drift detectors on the error signal.

### Q: How does GDPR affect ML systems in production?
Article 22 (right to explanation / human review for automated decisions) drives demand for explainability tooling (SHAP). Article 17 (right to be forgotten) requires removing user data from future training sets (or complex unlearning). Article 33 requires notifying regulators of data breaches within 72 hours (including training data / feature stores).

---

## Medium

### Q: What is ONNX, and why would you convert a PyTorch model to it before deployment?
A: **ONNX (Open Neural Network Exchange)** is an open standard format for representing machine learning models. A model trained in PyTorch or TensorFlow is heavily tied to the runtime of that specific framework (Python dependencies, eager execution overhead). Converting it to ONNX "traces" the computation graph and freezes it into a static, framework-agnostic protobuf file. This allows you to deploy the model in environments where Python isn't viable (like a C++ game engine, an edge device, or an iOS app) using a lightweight runtime (like ONNX Runtime). It also enables hardware-specific optimizations, as ONNX graphs can be easily compiled by hardware-specific backends (TensorRT for NVIDIA GPUs, OpenVINO for Intel, CoreML for Apple) without having to rewrite the model code.


> Round 2 depth — applied debugging, design trade-offs, serving optimizations, and "how would you build this?" questions.

### Q: Walk through the 7-step framework for an ML system design interview.
(1) Clarify requirements (objective, scale, latency); (2) Frame the ML problem (what are we predicting?); (3) Propose data sources and label collection (label delay/noise); (4) Design features (available at serving vs offline); (5) Propose model architecture (baseline first); (6) Design serving/system architecture (latency budget hops); (7) Define evaluation (offline metrics + online A/B guardrails).

### Q: Compare Lambda and Kappa architecture for ML data pipelines.
**Lambda**: maintains parallel batch (high latency, complete) and speed (low latency, streaming) pipelines. Costly to maintain two identical-logic codebases. **Kappa**: single stream-processing pipeline (Flink/Spark Streaming) for both real-time and historical processing (by replaying the event log). Kappa is the modern default, avoiding dual-pipeline skew.

### Q: When would you choose Delta Lake vs Apache Iceberg?
Both provide ACID transactions and time travel on object storage. **Delta Lake** has tighter integration with Databricks/Spark. **Iceberg** is open/engine-agnostic (Spark, Trino, Flink, Snowflake) and has superior partition evolution (change partitioning without rewriting all data) and better metadata scalability for massive tables.

### Q: What is point-in-time correctness in feature engineering, and why is a naive join wrong?
When constructing a training example for a label at time T, joined features must reflect only information available *before* T. A naive join grabs the *latest* feature value today, leaking future information into historical training examples (inflating offline metrics but failing in production). Use an as-of/temporal join to find the most recent value strictly before T.

### Q: Why do production retrieval systems use Approximate Nearest Neighbor (ANN) search?
Exact nearest neighbor search requires a dot product against every catalog item (O(N)), which is too slow (e.g., 30ms for 100M items) for a <10ms budget. ANN (HNSW, IVF-PQ) trades a small amount of recall (1-10%) for 100-1000x speedup by pre-organizing the embedding space.

### Q: What is Sample Ratio Mismatch (SRM) in experiments?
When the observed traffic split significantly diverges from the intended split (e.g., assigned 50/50 but observed 48/52). It's a red flag for an assignment/logging bug (e.g., treatment crashes for a specific browser, removing those users). Any experiment with SRM must be discarded, as measured differences could be pure selection bias.

### Q: Why is nDCG preferred over precision@k for search ranking?
Precision@k treats every position in top-k equally. Users pay logarithmically less attention to lower-ranked results. nDCG applies a position-dependent discount (dividing gain by log2(position+1)), rewarding models that put the highly relevant document at position 1 instead of position 5.

### Q: What is continuous batching in LLM serving?
Static batching waits for an entire batch of requests to finish before forming the next batch, wasting GPU compute on padded sequences. Continuous batching re-evaluates the batch at every single token-generation step: when a sequence finishes, its slot is immediately filled with a new request. Raises GPU utilization dramatically.

### Q: What is PagedAttention (vLLM)?
Traditional KV cache pre-allocates contiguous memory for the max sequence length, wasting memory. PagedAttention borrows OS virtual-memory paging: KV cache is divided into fixed-size physical blocks allocated on-demand as the sequence grows. Eliminates fragmentation, fitting 2-4x more concurrent requests in VRAM.

### Q: Why is training-serving skew one of the most dangerous failure modes?
Feature values computed at training differ from serving (e.g., batch nightly vs real-time Redis) due to dual codepaths. The model looks great offline but fails silently in production because predictions are quietly wrong based on skewed inputs. Fix: single feature definition compiled to both paths, and continuous online/offline parity testing.

### Q: What is the champion/challenger pattern vs canary?
Champion/challenger runs both models simultaneously on a stable traffic split for an extended comparison window. Champion serves the user; challenger is logged for offline comparison (measurement). Canary is explicitly a rollout mechanism with automated ramp-and-rollback logic tied to guardrails.

### Q: What is cascade serving?
Routing requests through progressively more expensive stages, short-circuiting when confidence is high. Stage 1: cheap rule-based logic (majority of traffic). Stage 2: lightweight quantized model. Stage 3: full expensive model (hardest 10%). Cuts average latency and cost by 5-10x while preserving accuracy on the hard tail.

### Q: What is a circuit breaker pattern in ML serving?
Wraps model/feature-store calls with a state machine (CLOSED, OPEN, HALF_OPEN). If failure rate exceeds a threshold, the circuit OPENS, short-circuiting calls immediately to a fallback (cached prediction, safe default) rather than waiting for a timeout. Prevents cascading failures.

### Q: How do you operationalize prediction bias / fairness monitoring?
Compute per-subgroup versions of primary metrics (FNR, FPR, calibration). Add explicit fairness gates in the deployment pipeline: a model cannot promote to production if a subgroup disparity exceeds a defined threshold, even if aggregate accuracy improved.

### Q: Why might "reframing" the ML problem be better than a more complex model?
Reframing changes *what* you predict. E.g., instead of predicting exact watch time (high variance regression), predict multiple classification buckets (short/medium/long). Instead of predicting "fraud" directly, predict auxiliary tasks (new device, new payee). The obvious target is often not the most robust one.

### Q: Pointwise vs pairwise vs listwise ranking loss?
**Pointwise**: predicts absolute relevance score per item independently (simple, ignores relative order). **Pairwise** (RankNet): trains on item pairs, optimizing so preferred items score higher (optimizes relative order). **Listwise** (LambdaMART): optimizes loss over the entire ranked list, approximating metrics like nDCG.

### Q: What signals indicate you need to trigger a model retrain?
(1) Drift magnitude crossing a threshold (PSI > 0.25) sustained over a window. (2) Measured drop in a delayed-label evaluation metric below a floor. (3) Scheduled cadence (e.g., monthly) as a backstop. Alert-only is appropriate if the signal is ambiguous or the fix is pipeline-related, not model-related.

### Q: You're told "the model's accuracy dropped." How do you respond?
"Accuracy" is the wrong metric for imbalanced classes (e.g., fraud). Ask what metric aligns with business impact (PR-AUC, recall at fixed FPR). Then ask for segment-level breakdown — an aggregate drop is often driven entirely by one segment (e.g., iOS clients) due to a broken feature.

### Q: How do you investigate feature store staleness?
Check scope: uniform (systemic backlog) vs user-specific (partition lag/dead letters). Check Kafka consumer lag on the streaming job. Check for dual-write consistency problems (online and offline written via separate paths, one failing). Check TTL configurations and fallback cache behavior.

### Q: What's the dual-write consistency problem?
Writing the same event to two stores (online Redis and offline Data Lake) independently. A partial failure leaves them permanently inconsistent. Solutions: CDC (Debezium) from a single source-of-truth database, Outbox pattern, or Event Sourcing.

### Q: Predicate pushdown vs partition pruning?
**Partition pruning**: skips entire partitions (date directories) based on query filters without opening files. **Predicate pushdown**: pushes filters into the file format's metadata (Parquet min/max stats per row-group) to skip blocks within a file. Fails if filters use complex UDFs or non-sargable functions (like `LIKE '%xyz'`).

### Q: Design a data quality validation gate for a training pipeline.
Sits right after ingestion. Checks (via Great Expectations): schema conformance, value range bounds, null-rate thresholds, distributional checks (vs a matched prior period to handle seasonality), and row-count anomaly detection. Fails fast rather than polluting the pipeline.

### Q: Streaming vs batch feature decision?
Tradeoff: freshness vs engineering cost. Delay the feature's refresh in offline evaluation and measure the target metric drop. If the drop is <0.5%, batch (hourly/nightly) is sufficient. Only use streaming (Flink/Kafka) when predictive value genuinely depends on recent events (e.g., last 5 minutes of clicks).

### Q: How do you handle incremental updates to a large ANN index?
Use a delta index: new items go to a small flat exact-search index. Queries search both the base ANN index and the delta index, merging top-K results. Periodically, the delta is merged into the base index in the background to prevent it from growing too large.

### Q: Why is ML experimentation harder than standard web A/B testing?
(1) Longer feedback loops (D30 retention). (2) Interference between users (e.g., marketplace supply/demand balance means treatment affects control). (3) Offline-online metric disconnect (offline AUC does not reliably predict online revenue).

### Q: When would you use a multi-armed bandit instead of A/B test?
Bandits (Thompson Sampling) dynamically shift traffic to the winning variant as evidence accumulates. Use when reward signal is immediate (click/purchase) and the cost of exploring the worse variant is high (real-time promotion ranking). Use A/B when feedback is delayed or you need a clean, unbiased causal estimate.

### Q: What is interleaving for ranking models?
Merges the ranked output of models A and B into a single list shown to the user, tracking which model contributed clicked items. Dramatically reduces between-user variance by comparing models in the same session, reaching significance much faster than standard A/B tests for ranking preference.

### Q: Fraud detection — why is a fixed precision/recall target wrong?
Error types have wildly different dollar costs (missing a $50k wire fraud vs blocking a $20 purchase). Use a cost-sensitive dynamic threshold chosen to minimize expected total cost, optimizing recall for high-value segments and precision for high-volume micro-transactions.

### Q: Ad CTR prediction — why does calibration matter more than AUC?
In real-time bidding, expected value = pCTR × bid. Absolute prediction value determines money spent. A model with great AUC (ranking) but 2x miscalibration will systematically over- or under-bid by 2x. Calibration is a first-class guardrail.

### Q: Two-tower vs joint (cross-encoder) model for recommendation?
**Two-tower**: encodes user and item independently. Item embeddings can be precomputed and indexed via ANN for fast O(log N) retrieval over millions of items. **Joint model**: allows deep feature interaction (attention) but must be run per user-item pair (O(N)), making it strictly a reranker for small candidate sets (hundreds).

