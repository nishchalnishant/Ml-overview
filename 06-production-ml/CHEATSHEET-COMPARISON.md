---
module: Production ML
topic: Rapid Review
subtopic: Comparison Cheat Sheet
status: unread
tags: [productionml, cheatsheet, comparison, interview-prep]
---
# Production ML — Comparison Cheat Sheet

Exhaustive, comparison-oriented rapid-review reference for `06-production-ml/`. Every entry grounded in what the folder's files actually cover (`01-mlops.md`, `02-deployment-patterns.md`, `03-model-governance.md`, `REVISION.md`, and `system-design/*.md`).

---

## Deployment Strategies

**Big-Bang / Rolling replacement**
- What it is: replace the old model/version entirely in one shot (or roll pods over gradually with no parallel comparison).
- Pros: simplest to operate; no extra infra.
- Cons: hard rollback (requires restart/redeploy); no live comparison before full exposure; blast radius = 100% of traffic.
- Pick over alternatives: only for internal tools / low-stakes models where a bad deploy is cheap to fix.
- Key operational detail: rollback is a redeploy, not a switch — recovery time is minutes, not seconds.

**Blue-Green**
- What it is: two full parallel environments (blue=current, green=new); traffic is atomically flipped from one to the other.
- Pros: instant rollback (flip traffic back); zero-downtime cutover; simple mental model.
- Cons: 2x infrastructure cost (both environments fully provisioned); doesn't catch issues that only appear under gradual load; no statistical comparison of variants.
- Pick over alternatives: pick blue-green over canary when you need **instant** rollback and can absorb 2x infra cost — e.g., high-stakes services where even a 5% canary failure is unacceptable exposure.
- Key operational detail: rollback time ~ seconds (traffic-switch only); cost = 2x steady-state infra during cutover window.

**Canary**
- What it is: route a small % of traffic (e.g., 5%) to the new version, monitor, then progressively ramp (5%→25%→50%→100%, often with 24h holds per step).
- Pros: cheap (no full duplicate environment); catches regressions on real traffic before full exposure; rollback = just reduce canary %.
- Cons: needs enough traffic volume for statistical power at low %; canary on a skewed segment (e.g., mobile-only) won't generalize; slower to reach full rollout than blue-green.
- Pick over alternatives: pick canary over blue-green when infra cost matters and gradual validation is acceptable; pick canary over shadow when you need to measure real user-facing impact (not just parallel scoring).
- Key operational detail: automated rollback triggers on error_rate / p99_latency / metric regression (e.g., PR-AUC drop, FP rate spike) — see Argo Rollouts progressive-steps pattern in `system-design/10-model-registry-versioning.md`.

**Shadow Deployment**
- What it is: new model runs in parallel on production traffic, computes predictions, but predictions are logged only — never served to users.
- Pros: zero user-facing risk; validates real-world performance/latency before any exposure; ideal for high-stakes models (credit, fraud, medical).
- Cons: 2x compute cost (both models score every request); doesn't measure actual user reaction/behavior change; requires infra to log and compare shadow vs production predictions.
- Pick over alternatives: pick shadow over canary when the cost of even 1% bad exposure is unacceptable (regulatory/high-stakes) — validate fully before any traffic hits the new model.
- Key operational detail: no rollback needed (nothing served); cost is purely compute-doubling, not user risk.

**A/B Testing**
- What it is: split traffic randomly (typically user-level, hashed for consistency) between variant A and B to get a causal, statistically-significant comparison of a metric.
- Pros: gives causal proof of impact, not just health-check monitoring; supports guardrail metrics alongside primary metric.
- Cons: requires power analysis/sample-size planning up front; slower to a decision than canary (needs statistical significance, not just "no errors"); vulnerable to peeking, SRM, novelty effects.
- Pick over alternatives: pick A/B over canary when the goal is proving a business-metric improvement (not just "did it break"); pick over bandits when feedback is delayed (e.g., D7 retention) and you need a clean, fixed-horizon causal read.
- Key operational detail: validate randomization with an A/A test before launch; use deterministic hashing for consistent user bucketing.

### Comparison Table

| Pattern | Best for | Avoid when | Key tradeoff |
|---|---|---|---|
| Big-Bang | Low-stakes internal tools | Any user-facing/high-stakes system | No rollback safety net |
| Blue-Green | High-stakes, need instant rollback | Cost-constrained infra | 2x infra cost for instant reversibility |
| Canary | Standard production deploys | Low-traffic services (no statistical power) | Cheap but slower full rollout |
| Shadow | Validate high-stakes models pre-exposure | Need to measure actual user behavior | 2x compute, zero user risk |
| A/B Test | Causal proof of metric impact | Need a fast go/no-go health check | Slower, but statistically defensible |

---

## Serving Patterns

**Batch / Async Inference**
- What it is: score large volumes of data on a schedule (nightly, hourly), write results to storage for later retrieval.
- Pros: very high throughput; cheap compute (no always-on serving infra); simplest to reason about correctness.
- Cons: predictions are stale between runs; unusable when freshness matters (fraud, bidding).
- Pick over alternatives: pick batch over real-time when prediction freshness doesn't matter (e.g., nightly churn scores) — static prediction is simpler and cheaper per `system-design/01-machine-learning-engineering.md`'s static-vs-dynamic framing.
- Key operational detail: cost multiplier ~1x baseline (vs 2-50x for real-time tiers).

**Real-time REST / Synchronous**
- What it is: model served behind a request/response API; caller waits for prediction inline.
- Pros: simple integration; good tooling/observability; human-readable payloads.
- Cons: higher per-request latency overhead (JSON serialization) vs gRPC; scales worse under very high QPS than binary protocols.
- Pick over alternatives: pick REST over gRPC when client diversity/debuggability matters more than raw latency.
- Key operational detail: typical latency tier <100ms (near-real-time) per `05-real-time-ml-systems.md` latency-tier table.

**Real-time gRPC**
- What it is: binary protocol (protobuf) request/response serving, typically used for internal service-to-service calls needing low latency.
- Pros: lower serialization overhead, smaller payloads, streaming support, faster than REST at high QPS.
- Cons: less human-debuggable; requires protobuf schema management; less universal client support.
- Pick over alternatives: pick gRPC over REST when P99 budget is tight (<10-50ms) and both sides are internal services.
- Key operational detail: used in <10ms synchronous tiers (ad bidding, fraud) where every millisecond of serialization counts.

**Streaming (Kafka + model)**
- What it is: model consumes events off a stream (Kafka) continuously, emits predictions as new events rather than responding to discrete requests.
- Pros: naturally event-driven; decouples producers/consumers; good for continuously-arriving data (feature computation, fraud velocity).
- Cons: added architectural complexity (partitioning, consumer groups, delivery semantics, watermarks for late data); harder to reason about exactly-once correctness.
- Pick over alternatives: pick streaming over request/response when the system reacts to a continuous flow of events rather than discrete user requests (e.g., real-time feature computation feeding a separate serving layer).
- Key operational detail: delivery semantics tradeoff — at-least-once (idempotent writes) is the pragmatic default; exactly-once (Kafka transactions) costs 2-3x latency overhead, reserve for financial ledgers.

**Edge Inference**
- What it is: model runs on-device (phone, IoT), no network round-trip to a server.
- Pros: zero network latency; works offline; privacy-preserving (data never leaves device).
- Cons: constrained compute/memory; harder to update (no central rollback); no centralized monitoring of live inputs.
- Pick over alternatives: pick edge over server-side serving when privacy or offline-availability is a hard requirement, and the model can be sufficiently compressed.
- Key operational detail: requires aggressive quantization/distillation to fit device constraints.

**LLM-specific serving: Continuous Batching, PagedAttention, Speculative Decoding** (see `system-design/12-llm-inference-ops.md`)
- What it is: continuous batching re-forms the GPU batch at every decode step (vLLM/TGI) instead of waiting for a full batch to finish; PagedAttention (vLLM) manages KV cache as fixed-size non-contiguous blocks via a block table, eliminating fragmentation; speculative decoding uses a small draft model to propose K tokens, verified in one parallel pass by the target model.
- Pros: continuous batching raises GPU utilization from ~30% (static batching) to >80%; PagedAttention gives 2-4x more concurrent requests (less wasted KV memory); speculative decoding gives 3-4x speedup on memory-bound decode.
- Cons: continuous batching requires custom scheduler complexity (preemption, admission control); PagedAttention needs block-table bookkeeping; speculative decoding needs a well-aligned draft model or acceptance rate is too low to help.
- Pick over alternatives: pick continuous batching over static batching whenever request lengths are heterogeneous (always true for chat/completion workloads); pick speculative decoding specifically when decode is memory-bandwidth-bound at low batch size (batch=1 serving).
- Key operational detail: decode is memory-bandwidth-bound (arithmetic intensity ~1 op/byte at batch=1) while prefill is compute-bound (intensity scales with sequence length) — this asymmetry is why batching decode requests matters so much more than batching prefill.

### Comparison Table

| Pattern | Best for | Avoid when | Key tradeoff |
|---|---|---|---|
| Batch | Non-real-time bulk scoring | Freshness matters | Cheapest, but stale |
| REST | User-facing APIs, debuggability | Ultra-low-latency internal calls | Simplicity vs raw speed |
| gRPC | Internal low-latency service calls | Need human-readable/debuggable payloads | Speed vs debuggability |
| Streaming (Kafka) | Continuous event-driven pipelines | Simple discrete request/response needs | Throughput vs architectural complexity |
| Edge | Privacy/offline requirements | Need centralized monitoring/fast updates | Latency/privacy vs compute constraints |

---

## Model Registry & Versioning

**MLflow**
- What it is: open-source experiment tracking + model registry with UI, lineage, serving, and run comparison.
- Pros: full feature set (UI, lineage, serving, comparison) per `system-design/10-model-registry-versioning.md`; works with any stack (not cloud-locked); free/open source.
- Cons: self-hosted/managed overhead; less turnkey than cloud-native options for teams already fully on one cloud.
- Pick over alternatives: pick MLflow over cloud-native registries when you need multi-cloud portability or are not committed to AWS/GCP.
- Key operational detail: supports full registration + stage-transition workflow (Staging → Production → Archived) and champion/challenger routing patterns.

**Weights & Biases (W&B) Registry**
- What it is: experiment tracking and model registry oriented at deep learning teams, with strong UI and comparison tooling.
- Pros: UI, lineage, comparison all strong; excellent for DL experiment visualization.
- Cons: no built-in serving support (✗ in comparison table) — must pair with separate serving infra.
- Pick over alternatives: pick W&B over MLflow when the team is DL-heavy and values experiment visualization/comparison UX over built-in serving.
- Key operational detail: must integrate a separate serving layer since W&B doesn't provide one.

**SageMaker Model Registry**
- What it is: AWS-native model registry integrated with SageMaker Pipelines for train/deploy/serve.
- Pros: UI, lineage, and serving all supported natively; tight AWS ecosystem integration (IAM, S3, endpoints).
- Cons: comparison tooling only "Partial"; locks you into AWS.
- Pick over alternatives: pick SageMaker MR over MLflow when already AWS-native and want managed serving without extra integration work.
- Key operational detail: best for AWS-native shops; portability cost if multi-cloud.

**Vertex AI Model Registry**
- What it is: GCP-native model registry integrated with Vertex AI Pipelines.
- Pros: UI, lineage, and serving supported; tight GCP integration.
- Cons: comparison tooling only "Partial"; GCP lock-in.
- Pick over alternatives: pick Vertex AI MR over MLflow when already GCP-native.
- Key operational detail: mirrors SageMaker MR tradeoffs but for GCP.

**Hugging Face Hub (as registry)**
- What it is: model hosting/versioning hub oriented around NLP/LLM model artifacts.
- Pros: serving supported (✓); huge ecosystem for pretrained NLP/LLM models.
- Cons: no lineage tracking (✗); no comparison tooling (✗) — not a full MLOps registry, more a model-sharing hub.
- Pick over alternatives: pick Hugging Face over MLflow/cloud registries specifically for NLP/LLM model distribution, not as your primary governance registry.
- Key operational detail: pair with MLflow/cloud registry if you need lineage + governance on top of HF-hosted weights.

### Comparison Table

| Tool | Best for | Avoid when | Key tradeoff |
|---|---|---|---|
| MLflow | Open source, any stack | Want fully managed, zero-ops | Most complete feature set, self-hosted |
| W&B Registry | Deep learning teams | Need built-in serving | Best comparison UX, no serving |
| SageMaker MR | AWS-native | Multi-cloud requirement | Managed serving, partial comparison |
| Vertex AI MR | GCP-native | Multi-cloud requirement | Managed serving, partial comparison |
| Hugging Face | NLP/LLM model distribution | Need lineage/governance | Great ecosystem, weak governance features |

---

## Feature Stores

**Offline Store**
- What it is: batch-correct historical feature storage (Parquet/Delta Lake/BigQuery/Hive) used to generate training data via point-in-time correct joins.
- Pros: supports point-in-time correctness (join features to labels using only pre-label-event data); scalable batch compute; reusable across teams.
- Cons: not usable for low-latency serving; requires a separate online store for inference.
- Pick over alternatives: pick offline-store-based training data generation over ad-hoc feature computation whenever multiple models/teams reuse the same features — prevents each team recomputing (and subtly redefining) the same feature.
- Key operational detail: point-in-time join must use `entity_df` + event timestamps (`as_of`); naive joins leak future data into training.

**Online Store**
- What it is: low-latency key-value store (Redis, DynamoDB, Bigtable, Cassandra) serving precomputed features at inference time.
- Pros: sub-5ms P99 feature lookups (Redis ~0.5ms p50/2ms p99); enables real-time model serving.
- Cons: RAM-bound capacity (Redis) or higher latency for larger/colder data (Cassandra/Bigtable); needs a write path (streaming) to stay fresh.
- Pick over alternatives: pick Redis for hot, small, high-QPS features; pick Cassandra/DynamoDB/Bigtable for large embeddings or cold entities where a few extra ms is acceptable.
- Key operational detail: latency figures — Redis 0.5ms p50/2ms p99; DynamoDB DAX 1ms/5ms; Bigtable 3ms/10ms; PostgreSQL+pgpool 5ms/20ms.

**Ad-hoc feature computation (no feature store)**
- What it is: each model/service computes its own features independently (training pipeline and serving pipeline separately implement the same logic).
- Pros: no extra infra to build/operate; fastest to prototype a single model.
- Cons: training-serving skew risk is high (two independent implementations of "the same" feature drift apart); no reuse across teams; no point-in-time correctness guarantee by default.
- Pick over alternatives: acceptable only for a single, simple model with no reuse need and low stakes; pick a feature store instead once >1 team/model shares features or skew risk is unacceptable.
- Key operational detail: the classic "user_30day_spend computed differently in batch vs real-time" failure mode described in `README.md`'s Train-Serve Skew section originates here.

**Feature Store Tools — Feast / Tecton / Hopsworks / Vertex AI FS / SageMaker FS**
- What it is: open-source (Feast) vs managed (Tecton, Hopsworks, Vertex AI FS, SageMaker FS) feature store platforms, differing in offline/online store options, streaming support, and deployment model.
- Pros: Feast is free/flexible/pluggable (bring your own offline+online store); managed options (Tecton/Hopsworks/cloud-native) reduce operational burden and include streaming ingestion built in.
- Cons: Feast requires you to operate the online/offline stores yourself; managed options cost more and can lock you to a cloud/vendor.
- Pick over alternatives: build (Feast) vs buy (Tecton/managed) threshold is roughly >100TB feature data or >100M lookups/day — below that, Feast is usually sufficient; above it, managed platforms' operational tooling pays for itself.
- Key operational detail: validate online/offline parity continuously (KS statistic comparing online-served vs offline-computed feature distributions) — silent parity drift is a common failure mode.

### Comparison Table

| Approach | Best for | Avoid when | Key tradeoff |
|---|---|---|---|
| Offline store (batch) | Training data generation, reuse | Low-latency serving needs | Correctness vs latency |
| Online store (Redis/DynamoDB/etc.) | Real-time feature lookup | Very large cold datasets (use Cassandra/Bigtable instead of Redis) | Latency vs capacity/cost |
| Ad-hoc computation | Single simple model, prototyping | Multiple teams/models share features | Fast to start, high skew risk |
| Feast (OSS) | <100TB / <100M lookups/day, flexible stack | Need managed streaming ingestion out of the box | Free but self-operated |
| Managed (Tecton/Hopsworks/cloud FS) | >100TB or >100M lookups/day | Cost-sensitive, small scale | Turnkey but vendor cost/lock-in |

---

## Monitoring & Drift Detection

**PSI (Population Stability Index)**
- What it is: statistical measure comparing a feature's distribution between a baseline (training) window and a current (serving) window.
- Pros: simple, single interpretable number; works well on binned continuous or categorical features; industry-standard threshold conventions.
- Cons: sensitive to binning choice; doesn't diagnose *why* a distribution shifted, only that it did.
- Pick over alternatives: pick PSI over KL divergence when you want a standard, threshold-driven trigger for retraining decisions (PSI > 0.25 = significant drift → retrain, per `REVISION.md`).
- Key operational detail: PSI < 0.1 = no significant shift; 0.1-0.25 = moderate, investigate; >0.25 = significant, retrain.

**KL Divergence**
- What it is: information-theoretic measure of how one probability distribution diverges from a reference distribution.
- Pros: theoretically grounded; sensitive to distributional shape differences beyond simple binning.
- Cons: asymmetric (KL(P||Q) ≠ KL(Q||P)); less standardized "actionable threshold" convention than PSI in practice; harder to explain to non-technical stakeholders.
- Pick over alternatives: pick KL divergence over PSI when you need a more theoretically rigorous comparison and have the statistical sophistication to interpret it; PSI is more common as an operational alerting metric.
- Key operational detail: used alongside PSI in skew-detection design pattern (`system-design/02-machine-learning-design-patterns.md` pattern #18).

**Kolmogorov-Smirnov (KS) Test**
- What it is: non-parametric statistical test comparing two continuous distributions (max distance between empirical CDFs).
- Pros: distribution-free (no binning needed, unlike PSI); gives a p-value for statistical significance; well-suited to continuous features.
- Cons: less sensitive to differences in the tails; not naturally suited to categorical features (use chi-squared instead).
- Pick over alternatives: pick KS over PSI for continuous features where you want a formal significance test rather than a heuristic index; pick chi-squared instead for categorical drift.
- Key operational detail: also used for online/offline feature-store parity testing (comparing online-served vs offline-computed feature distributions).

**ADWIN / DDM / Page-Hinkley (online/streaming drift detectors)**
- What it is: sequential statistical algorithms that detect concept drift in a streaming setting without needing to store the full history (ADWIN = adaptive windowing; DDM = drift detection method tracking error-rate increase; Page-Hinkley = cumulative sum change-point detection).
- Pros: designed for continuous/online learning settings; detect drift incrementally without batch recomputation; ADWIN adapts window size automatically.
- Cons: more complex to tune (thresholds, window parameters) than a scheduled batch PSI check; primarily suited to online/incremental learning pipelines, overkill for batch-retrained systems.
- Pick over alternatives: pick these over PSI/KS when the model is continuously updated online (`partial_fit`) rather than periodically batch-retrained — batch PSI checks are insufficient because there's no natural "batch boundary" to compare against.
- Key operational detail: relevant specifically to online/incremental learning architectures (risk: catastrophic forgetting if drift response over-corrects).

**Data Drift vs Concept Drift vs Label Drift vs Upstream Drift**
- What it is: four distinct failure modes — data drift = P(X) shifts (e.g. users shift to mobile); concept drift = P(Y|X) shifts (e.g. "good credit" definition changes); label drift = P(Y) shifts (e.g. fraud rate spikes seasonally); upstream drift = feature pipeline/schema changes silently.
- Pros of separating them: each has a different fix — data/label drift often just needs retraining or reweighting; concept drift needs retraining on recent data (old labels are now wrong); upstream drift needs schema validation, not retraining at all.
- Cons if conflated: teams that only monitor input distributions (data drift) miss concept drift entirely — accuracy can collapse while feature distributions look identical (explicitly called out as a fraud-model interview scenario in `system-design/05-real-time-ml-systems.md`).
- Pick over alternatives: diagnose concept drift specifically when feature distributions are stable but model accuracy/precision drops — the fix is retraining on recent labels, not a data pipeline fix.
- Key operational detail: upstream drift is caught by schema validation + alerting, not statistical distribution tests.

### Comparison Table

| Technique | Best for | Avoid when | Key tradeoff |
|---|---|---|---|
| PSI | Standard retraining trigger, categorical/binned | Need to diagnose exact shift shape | Simple, threshold-driven |
| KL Divergence | Rigorous distributional comparison | Need a standardized alert threshold | More rigorous, less operational convention |
| KS Test | Continuous feature drift, online/offline parity | Categorical features (use chi-squared) | Distribution-free, gives p-value |
| ADWIN/DDM/Page-Hinkley | Online/streaming incremental learning | Batch-retrained systems (overkill) | Adaptive, but harder to tune |

---

## Experimentation: A/B Testing vs Bandits vs Other Designs

**Classic A/B Testing**
- What it is: fixed-horizon randomized experiment; assign users deterministically (hashed) to variants, run until pre-computed sample size is reached, analyze with a statistical test.
- Pros: clean causal inference; well-understood statistics; supports guardrail metrics.
- Cons: "wastes" traffic on the losing variant for the full experiment duration; slow when the effect is time-delayed (e.g., D7 retention takes 7+ days to read out).
- Pick over alternatives: pick A/B over bandits when the outcome metric has delayed feedback (retention, LTV) — bandits need fast reward signal to adapt, which delayed metrics don't provide.
- Key operational detail: requires power analysis/sample-size calculation before launch; validate randomization via A/A test; detect Sample Ratio Mismatch (SRM) via chi-squared test on actual vs expected traffic split.

**Multi-Armed Bandits (e.g., Thompson Sampling)**
- What it is: adaptive traffic allocation that shifts more traffic toward better-performing variants *during* the experiment, rather than a fixed 50/50 split for the full duration.
- Pros: minimizes "cost of exploration" (less traffic wasted on the losing arm); good when the business cost of serving a suboptimal variant is high and reward is immediate.
- Cons: harder to get a clean, interpretable "effect size" for reporting; complicates statistical inference vs a clean fixed-horizon A/B test.
- Pick over alternatives: pick bandits over A/B testing when reward feedback is immediate (e.g., click, immediate conversion) and the cost of showing users a worse variant is high — per the explicit "Bandit vs A/B: when to use each" guidance in `system-design/14-ab-testing-experimentation.md`.
- Key operational detail: Thompson Sampling maintains a posterior over each arm's reward probability and samples from it to decide allocation — naturally balances explore/exploit.

**CUPED (variance reduction)**
- What it is: technique that uses pre-experiment covariate data to reduce variance in the treatment-effect estimate, increasing statistical power without more traffic.
- Pros: detects smaller effects with the same sample size; doesn't require running the experiment longer.
- Cons: needs good pre-period covariate data; adds analysis complexity.
- Pick over alternatives: pick CUPED when you're traffic-constrained and need more sensitivity from an existing experiment rather than waiting longer or bucketing more users.
- Key operational detail: implemented in `cuped_analysis` per the file — reduces variance using a correlated pre-experiment metric as a control variate.

**Interference / Network-effect designs (Cluster randomization, Geographic/Market split, Switchback testing, Hold-out group)**
- What it is: alternative randomization unit used when standard user-level A/B testing violates the "no interference" (SUTVA) assumption — e.g., marketplace/two-sided effects, network effects between users.
- Pros: cluster randomization and geo-splits avoid contamination between treatment/control when users interact with each other; switchback testing (time-based on/off) is useful when clusters are hard to define; hold-out groups measure long-run cumulative effect.
- Cons: cluster/geo-based designs increase required sample size and complexity (fewer independent units = less power); switchback introduces time-based confounds (day-of-week effects).
- Pick over alternatives: pick cluster/geo randomization over standard user-level A/B when users interact with each other (marketplace liquidity, social feed) so treatment leaks into control; pick switchback when clusters aren't naturally definable but time-based on/off is feasible; pick hold-out groups to measure cumulative long-term effect that a short experiment window would miss.
- Key operational detail: complexity increases roughly in the order: user-level A/B < geographic split < cluster randomization < switchback (per the table in `14-ab-testing-experimentation.md`).

**Interleaving (Team-Draft Interleaving)**
- What it is: for ranking/search comparison, interleave results from two rankers into a single result list shown to the user, then attribute clicks back to the source ranker.
- Pros: far more sample-efficient than A/B testing for ranker comparison — detects preference with much less traffic since every impression is a paired comparison.
- Cons: only applicable to ranking/list-based outputs, not general model comparison; more complex to implement (interleaving logic + winner attribution).
- Pick over alternatives: pick interleaving over standard A/B specifically for ranker/search comparisons where sample efficiency matters and outputs are ordered lists.
- Key operational detail: winner is computed by counting which ranker contributed more clicked results in the interleaved list.

**Common Pitfalls table** (from `14-ab-testing-experimentation.md`)

| Pitfall | Problem | Fix |
|---|---|---|
| Peeking | Checking significance repeatedly inflates false-positive rate | Fix sample size/duration up front, don't stop early |
| Multiple comparisons | Testing many metrics/segments inflates false positives | Correct for multiple testing (Bonferroni etc.) or pre-register primary metric |
| Novelty effect | Users react to "new" not "better" | Run long enough for novelty to fade |
| Primacy effect | Existing users resist change, initial effect underestimates true effect | Run long enough / segment new vs existing users |
| SRM (Sample Ratio Mismatch) | Actual traffic split deviates from intended (bug in assignment) | Detect via chi-squared test on split before trusting results |
| Practical vs statistical significance | Statistically significant but business-irrelevant effect size | Report effect size + confidence interval, not just p-value |

### Comparison Table

| Design | Best for | Avoid when | Key tradeoff |
|---|---|---|---|
| A/B Test | Delayed-feedback metrics, clean causal read | Need fast adaptive optimization | Clean inference, slower/wastes traffic on losing arm |
| Multi-Armed Bandit | Immediate reward, high cost of bad variant | Need clean statistical effect-size reporting | Less traffic wasted, harder to report cleanly |
| CUPED | Traffic-constrained, need more power | No good pre-period covariate available | More sensitivity, more analysis complexity |
| Cluster/Geo/Switchback | Network effects/interference present | Effects are user-independent (plain A/B suffices) | Avoids contamination, costs statistical power |
| Interleaving | Ranker/search comparison | Non-ranking model comparisons | High sample efficiency, narrow applicability |

---

## ML Design Patterns

(From `system-design/02-machine-learning-design-patterns.md` — 20 patterns across 4 groups.)

### Data Representation

**Hashed Feature**
- What it is: hash high-cardinality categorical values into a fixed-size bucket space instead of maintaining an ever-growing vocabulary.
- Pros: bounded memory regardless of cardinality growth; handles unseen categories at serving time gracefully (no "unknown token" crash).
- Cons: hash collisions merge distinct categories, adding noise; loses interpretability of individual categories.
- Pick over alternatives: pick hashing over a fixed vocabulary/one-hot when cardinality is unbounded or unknown at training time (e.g., user IDs, URLs).
- Key operational detail: bucket count controls the collision-vs-memory tradeoff — more buckets = fewer collisions, more memory.

**Embeddings**
- What it is: learned dense vector representation of high-cardinality categorical or discrete entities.
- Pros: captures similarity between entities; drastically smaller than one-hot for high cardinality; transferable/reusable across models.
- Cons: requires enough data per entity to learn a meaningful representation; cold-start problem for new entities; adds training complexity.
- Pick over alternatives: pick embeddings over hashed features when interpretable similarity structure matters and there's enough data to learn it (vs hashing, which is just collision-tolerant bucketing with no learned structure).
- Key operational detail: cold-start entities need a fallback (default/average embedding) until enough interaction data accumulates.

**Feature Cross**
- What it is: combine two or more features into a new composite feature to capture interaction effects a linear model can't represent alone.
- Pros: lets simple/linear models capture non-linear interactions cheaply; interpretable if crosses are chosen deliberately.
- Cons: combinatorial explosion in cardinality (cross of two high-cardinality features is huge); needs hashing to bound size.
- Pick over alternatives: pick feature crosses over relying on a deep model to learn interactions when using a simpler/linear model architecture or when specific known interactions matter (e.g., city x day-of-week).
- Key operational detail: often paired with Hashed Feature pattern to bound the resulting cardinality.

**Multimodal Input**
- What it is: combine heterogeneous input types (text, image, tabular, categorical) into a single model.
- Pros: leverages complementary signal across modalities; single unified model rather than separate pipelines.
- Cons: requires modality-specific preprocessing/encoders; harder to debug which modality drives a prediction; heavier compute.
- Pick over alternatives: pick multimodal fusion when a single modality alone is insufficient (e.g., product image + description + price for a recommendation model).
- Key operational detail: typically encode each modality separately (e.g., CNN for image, embedding for text) then concatenate/fuse before the shared head.

### Problem Representation

**Reframing**
- What it is: change the problem type (e.g., regression → classification into buckets) when the reframed version is easier to model or better matches business needs.
- Pros: can dramatically simplify learning (bucketed classification often outperforms exact regression when precision beyond the bucket isn't needed); aligns loss function with actual business decision.
- Cons: loses granularity within a bucket; bucket boundaries are an added design decision.
- Pick over alternatives: pick reframing over forcing a hard regression problem when the downstream decision only needs a coarse bucket (e.g., "will this take 0-1 day, 1-3 days, or 3+ days" instead of exact hours).
- Key operational detail: directly follows from the Problem Framing principle in `01-machine-learning-engineering.md` — translate the business decision into the ML task, not the reverse.

**Multilabel Classification**
- What it is: predict multiple non-mutually-exclusive labels per example (vs single-label multiclass).
- Pros: models real-world cases where multiple labels genuinely co-occur (e.g., a document with multiple topics).
- Cons: standard accuracy metrics don't apply cleanly; requires per-label thresholding decisions.
- Pick over alternatives: pick multilabel over multiclass when labels aren't mutually exclusive in the domain.
- Key operational detail: evaluate with per-label precision/recall or micro/macro F1, not plain accuracy.

**Ensembles (bagging/boosting/stacking)**
- What it is: combine multiple models' predictions to improve accuracy/robustness over any single model.
- Pros: bagging reduces variance (e.g., random forest); boosting reduces bias sequentially (e.g., XGBoost); stacking combines heterogeneous model types via a meta-learner.
- Cons: increased serving complexity/latency (multiple models to run); harder to interpret and debug than a single model.
- Pick over alternatives: pick bagging when variance/overfitting is the main problem; boosting when bias/underfitting is the main problem; stacking when you have several diverse strong models and want to combine their complementary strengths.
- Key operational detail: ensemble serving latency = sum (or max, if parallelized) of constituent model latencies — factor into latency budget.

**Cascade**
- What it is: chain of models of increasing cost/accuracy, where only ambiguous cases proceed to the next (more expensive) stage.
- Pros: dramatically cuts average latency/cost since most traffic resolves at cheap early stages (explicit example in `13-cost-optimization.md`: 50% rules + 35% fast model + 15% full model → 81.5% cost reduction vs always using the full model).
- Cons: added system complexity (multiple models, routing logic, confidence thresholds to tune); errors at an early stage can't be corrected downstream if routed incorrectly.
- Pick over alternatives: pick cascade over always running the expensive model when a large fraction of traffic is "easy" and can be resolved cheaply/confidently by a lightweight first stage.
- Key operational detail: confidence thresholds for escalation must be tuned carefully — too conservative negates the cost savings, too aggressive sends hard cases through with wrong early answers.

**Neutral Class**
- What it is: add an explicit "neutral/uncertain" class rather than forcing a binary decision on ambiguous examples.
- Pros: prevents the model from being forced into confident wrong answers on genuinely ambiguous inputs; downstream logic can route neutral cases to human review.
- Cons: adds a class to manage in training data and evaluation; downstream systems must handle a three-way (not two-way) decision.
- Pick over alternatives: pick a neutral class over forcing binary classification when a meaningful fraction of real examples are genuinely ambiguous (e.g., borderline moderation content).
- Key operational detail: pairs naturally with human-in-the-loop review queues for the neutral bucket.

**Rebalancing (class weighting / SMOTE / threshold adjustment)**
- What it is: techniques to address severe class imbalance — reweight the loss function, synthetically oversample the minority class (SMOTE), or adjust the decision threshold post-training.
- Pros: class weighting and threshold adjustment are cheap and don't distort the underlying data; SMOTE can help when there's too little minority-class signal for the model to learn at all.
- Cons: SMOTE can generate unrealistic synthetic examples in high-dimensional space; naive oversampling risks overfitting to duplicated minority examples; threshold adjustment alone doesn't fix a poorly-calibrated model.
- Pick over alternatives: pick threshold adjustment first (cheapest, no retraining) over resampling when the model already separates classes reasonably but the default 0.5 threshold is wrong for the business cost asymmetry; pick class weighting/SMOTE when the model genuinely can't learn the minority class pattern at all.
- Key operational detail: always evaluate on the natural (non-rebalanced) class distribution — rebalancing changes training data/loss, not the real-world prior.

### Model Training

**Transform**
- What it is: encapsulate feature transformation logic (scaling, encoding) so it travels with the model and is guaranteed identical between training and serving.
- Pros: directly prevents training-serving skew from mismatched preprocessing — the single biggest production ML failure mode per this repo's README.
- Cons: adds a layer of abstraction/tooling (must serialize and version the transform alongside the model).
- Pick over alternatives: pick an explicit Transform pattern over "reimplement preprocessing in the serving service" whenever training and serving are different codebases/languages.
- Key operational detail: fit scalers/encoders only on training data, never on val/test/serving data — same principle repeated across `01-machine-learning-engineering.md`'s data leakage section.

**Multistage Training**
- What it is: break training into sequential stages (e.g., pretrain on a broad task, then fine-tune on the target task) rather than training end-to-end from scratch.
- Pros: leverages cheaper/more abundant data in early stages; often converges faster and generalizes better than single-stage training from scratch.
- Cons: more pipeline complexity (multiple training jobs, checkpoint management between stages).
- Pick over alternatives: pick multistage training over single-stage when a related, larger dataset exists for pretraining and the target task's labeled data is scarce.
- Key operational detail: overlaps conceptually with Transfer Learning but multistage can also mean stage-wise curriculum on the *same* dataset (e.g., coarse-to-fine).

**Transfer Learning**
- What it is: reuse a model (or its learned representations) trained on one task/dataset as the starting point for a different but related task.
- Pros: drastically reduces data/compute needed for the target task; often outperforms training from scratch when target data is limited.
- Cons: risk of negative transfer if source and target domains are too dissimilar; may carry over unwanted biases from the source dataset.
- Pick over alternatives: pick transfer learning over training from scratch whenever labeled target data is scarce relative to what a from-scratch model would need.
- Key operational detail: typical practice — freeze early layers, fine-tune later layers on target data (or fully fine-tune with a lower learning rate).

**Distillation**
- What it is: train a smaller "student" model to mimic a larger "teacher" model's outputs (soft labels), compressing capability into a cheaper model.
- Pros: 3-10x inference speedup/cost reduction for typically 2-5% accuracy loss (per `13-cost-optimization.md` cost/quality tables); production-viable model that retains most teacher performance.
- Cons: added training pipeline step (train teacher, generate soft labels, train student); some accuracy loss is inevitable.
- Pick over alternatives: pick distillation over serving the large model directly when inference cost/latency at scale dominates and a small accuracy loss is acceptable; pick over quantization alone when you need a fundamentally smaller architecture, not just lower-precision weights.
- Key operational detail: often combined with quantization for compounding cost savings (distill then quantize).

**Regularization (L1/L2/Dropout/Early Stopping)**
- What it is: techniques constraining model complexity to prevent overfitting — L1/L2 penalize weight magnitude, dropout randomly zeroes activations during training, early stopping halts training when validation loss stops improving.
- Pros: L1 induces sparsity (implicit feature selection); L2 is smooth and generally stabilizing; dropout is cheap and effective for neural nets; early stopping requires no architecture change at all.
- Cons: adds a hyperparameter to tune (regularization strength, dropout rate, patience); overly aggressive regularization underfits.
- Pick over alternatives: pick L1 over L2 when feature selection/sparsity is desirable; pick dropout specifically for neural networks with enough capacity to overfit; pick early stopping as the lowest-effort default guard against overfitting in any iterative training setup.
- Key operational detail: always tune regularization strength against a held-out validation set, never the test set.

### Model Evaluation

**Evaluation Metrics (PR-AUC, NDCG, F-beta)**
- What it is: choosing task-appropriate metrics beyond plain accuracy — PR-AUC for imbalanced classification, NDCG for ranking quality, F-beta to weight precision vs recall by business cost.
- Pros: aligns the offline metric with the actual business objective (vs accuracy, which is misleading under imbalance).
- Cons: requires deliberate metric selection per task — no one-size-fits-all default.
- Pick over alternatives: pick PR-AUC over ROC-AUC under severe class imbalance (ROC-AUC is optimistic when negatives vastly outnumber positives); pick NDCG for ranking/recommendation where position matters; pick F-beta (beta>1 for recall-weighted, <1 for precision-weighted) when precision/recall costs are asymmetric.
- Key operational detail: mirrors the "97% precision at 30% recall passes evaluation but is commercially worthless" failure mode called out in `01-machine-learning-engineering.md`.

**Slicing**
- What it is: evaluate model performance on meaningful subpopulations/segments, not just the aggregate metric.
- Pros: catches subgroup failures hidden by a good aggregate score (explicitly flagged as a "common failure" in `REVISION.md`'s MLOps checklist: "Aggregate metric hides subgroup failure").
- Cons: requires defining meaningful slices up front (demographic, device type, geography); more evaluation infrastructure/reporting surface.
- Pick over alternatives: pick sliced evaluation over aggregate-only whenever fairness, safety, or known-heterogeneous-population risk exists.
- Key operational detail: pair with Prediction Bias/Fairness pattern for slices defined by protected attributes.

**Skew Detection (PSI/KL)**
- What it is: apply the same PSI/KL drift-detection statistics (see Monitoring section above) specifically to compare training-time vs serving-time feature distributions, catching training-serving skew before it causes a silent accuracy drop.
- Pros: proactively catches skew rather than waiting for a downstream accuracy/business-metric drop.
- Cons: needs logged serving-time feature snapshots to compare against training distributions.
- Pick over alternatives: pick this over waiting for delayed ground-truth labels to reveal a problem — feature-level skew detection is an earlier warning signal than label-based metrics.
- Key operational detail: directly ties to the "log features at serving time, use that logged snapshot for training" practice from `05-real-time-ml-systems.md`.

**Baseline Comparison**
- What it is: always compare the trained model against simple baselines (majority class, rule-based, plain logistic regression) before deploying.
- Pros: cheap sanity check; prevents deploying complex models that don't actually beat a trivial baseline; directly informs the project-viability/cost-vs-accuracy tradeoff.
- Cons: requires discipline to maintain baseline implementations alongside the "real" model over time.
- Pick over alternatives: mandatory regardless of alternatives — "if the ML model can't beat these, it has no business being deployed" (`01-machine-learning-engineering.md`).
- Key operational detail: baselines should be evaluated on the identical test set/metric as the candidate model, not a different slice.

**Prediction Bias / Fairness (demographic parity, equal opportunity, disparate impact, adversarial debiasing)**
- What it is: formal fairness metrics and mitigation techniques — demographic parity (equal positive-prediction rate across groups), equal opportunity (equal true-positive rate across groups), disparate impact ratio, and adversarial debiasing (train against an adversary trying to predict the protected attribute from the model's representation).
- Pros: makes fairness measurable and auditable rather than assumed; adversarial debiasing actively removes protected-attribute signal from learned representations.
- Cons: fairness constraints often trade off against raw accuracy (explicit Pareto tradeoff noted in `05-real-time-ml-systems.md`'s multi-objective section); different fairness definitions (demographic parity vs equal opportunity) can be mutually incompatible — satisfying one can violate another.
- Pick over alternatives: pick demographic parity when equal outcome rates across groups is the legal/policy requirement; pick equal opportunity when equal error rates for the qualified population matter more than equal outcome rates; pick adversarial debiasing when you need to remove protected-attribute signal from representations directly, not just adjust the decision threshold per group.
- Key operational detail: this is also where model governance's regulatory compliance obligations bite (GDPR Art 22 automated-decision rights, ECOA/FCRA in credit) — see `03-model-governance.md`.

---

## Model Governance & Compliance (from `03-model-governance.md`)

**Semantic Versioning for models (MAJOR.MINOR.PATCH)**
- What it is: version model artifacts with the same MAJOR (breaking change, e.g. new input schema)/MINOR (retrain, same schema)/PATCH (config/threshold tweak) semantics as software.
- Pros: communicates blast-radius of a model update at a glance; integrates with existing release tooling/conventions.
- Cons: requires discipline to classify every change correctly.
- Pick over alternatives: pick semantic versioning over ad-hoc naming/dates whenever multiple consumers depend on model input/output contracts.
- Key operational detail: MAJOR version bump should trigger the full 4-gate approval process; PATCH may be fast-tracked.

**4 Approval Gates + Audit Trails**
- What it is: staged sign-off process a model must pass before production (e.g., data/eval review, technical review, business review, compliance review), with a full audit trail of who approved what and when.
- Pros: catches issues before production exposure; provides regulatory-defensible documentation trail.
- Cons: adds latency to deployment; can become a rubber-stamp if not enforced meaningfully.
- Pick over alternatives: mandatory for regulated/high-stakes domains (credit, healthcare, insurance) where GDPR/ECOA/FCRA/HIPAA/Solvency II compliance is required; lighter-weight for low-stakes internal models.
- Key operational detail: audit trail must capture model version, data version, approver identity, and timestamp for each gate.

**Model Cards**
- What it is: standardized documentation artifact describing a model's intended use, training data, performance across slices, and known limitations.
- Pros: makes model capabilities/limitations discoverable to downstream consumers and auditors without reading code.
- Cons: another artifact to keep in sync with the actual deployed model — stale model cards are worse than none.
- Pick over alternatives: pick model cards over relying on tribal knowledge whenever a model is consumed by teams other than the one that built it, or is subject to external audit.
- Key operational detail: should embed sliced evaluation metrics, directly tying to the Slicing design pattern above.

**Incident Response (Severity P1-P4, rollback playbook)**
- What it is: predefined severity tiers and a documented rollback playbook for responding to production model incidents.
- Pros: removes decision-paralysis during an incident; ensures rollback (not just root-cause investigation) happens fast.
- Cons: requires the rollback mechanism (e.g., champion/challenger router, canary infra) to actually exist and be tested beforehand.
- Pick over alternatives: essential wherever automated rollback triggers (per `10-model-registry-versioning.md`'s `trigger_automatic_rollback`) are in place — the playbook is the human-process complement to the automated trigger.
- Key operational detail: P1 = full outage/harm, requires immediate rollback; lower severities may allow monitoring before action.

---

## Cost Optimization Techniques (from `system-design/13-cost-optimization.md`)

**Spot/Preemptible Instances**
- What it is: use interruptible cloud compute at 60-90% discount for fault-tolerant training jobs, with checkpointing to resume after preemption.
- Pros: 70-90% cost savings vs on-demand for training.
- Cons: requires fault-tolerant training code (frequent checkpointing) or you lose progress on interruption.
- Pick over alternatives: pick spot over on-demand for any training job that can checkpoint/resume; pick on-demand only for short, critical, interruption-intolerant jobs.
- Key operational detail: combine with reserved instances for a hybrid strategy (~25-40% cost vs 100% on-demand baseline).

**Quantization (INT8/INT4) vs Distillation vs Cascade Serving — cost angle**
- What it is: three different levers for cutting inference cost — quantization reduces numeric precision of an existing model; distillation trains a smaller model; cascade serving routes most traffic to cheap early stages.
- Pros: quantization is lowest-effort (often no retraining, PTQ) for 2-4x speedup at <1% accuracy loss; cascade serving gave an explicit 81.5% cost reduction example with minimal accuracy compromise (rules + fast model handle 85% of traffic).
- Cons: quantization alone doesn't reduce architecture size/FLOPs as much as distillation; cascade adds routing/threshold-tuning complexity; INT4 quantization (LLMs) can cost 1-3% accuracy.
- Pick over alternatives: pick quantization first (cheapest, least effort) before considering distillation or cascade; pick cascade when a large fraction of traffic is "easy" and can be resolved by a cheap early stage; pick distillation when you need a genuinely smaller model architecture, not just lower precision.
- Key operational detail: always measure cost per quality unit (cost per AUC/BLEU point), not absolute cost — the file explicitly warns against quantizing to the point where lost accuracy costs more (e.g., in missed fraud) than the compute saved.

**Result Caching**
- What it is: cache inference results keyed by a hash of input features, with a TTL, to avoid recomputation for repeated/similar requests.
- Pros: can achieve 40-70% hit rates for use cases with repeated queries (feed/search); near-zero cost for cache hits.
- Cons: not viable for real-time-critical scores that must reflect the absolute latest state (e.g., fraud — the file explicitly says "do NOT cache").
- Pick over alternatives: pick caching over recomputation whenever the same input recurs frequently and slight staleness (TTL window) is acceptable; avoid entirely when the score must be real-time-accurate every call.
- Key operational detail: TTL choice trades staleness vs hit rate — 5min TTL for personalized feed (~40% hit rate), 1h TTL for popular search queries (~70% hit rate).

### Comparison Table

| Technique | Best for | Avoid when | Key tradeoff |
|---|---|---|---|
| Spot instances | Fault-tolerant training | Short critical jobs that can't checkpoint | 70-90% savings, interruption risk |
| Quantization | Any model, first lever to try | Already at accuracy floor | Cheap, small accuracy loss |
| Distillation | Need smaller architecture, not just lower precision | No budget for teacher+student training pipeline | Bigger latency win, more setup cost |
| Cascade serving | Large fraction of "easy" traffic | Uniformly hard traffic (no easy cases to filter) | Big average-cost cut, added routing complexity |
| Result caching | Repeated/similar queries, staleness tolerable | Must-be-real-time scores (fraud) | High hit-rate savings, staleness risk |

---

## Real-Time Systems: Consistency & Failure Handling (from `system-design/05-real-time-ml-systems.md`)

**CDC (Change Data Capture) vs Event Sourcing vs Saga — dual-write consistency**
- What it is: three patterns solving the dual-write problem (must atomically update both a feature store and a message queue) — CDC reads the DB write-ahead log and publishes derived events; Event Sourcing treats the message queue itself as the single source of truth; Saga uses compensating actions across a multi-step distributed transaction.
- Pros: CDC gives a single DB source of truth with atomic guarantees at the DB level; Event Sourcing gives a natural audit log and simpler mental model; Saga avoids needing a distributed transaction coordinator.
- Cons: CDC adds WAL-replication lag (~100-500ms) and Debezium operational burden; Event Sourcing is only eventually consistent (Redis may lag by seconds); Saga adds compensation-logic complexity and a window of inconsistency during rollback.
- Pick over alternatives: pick CDC for financial ledgers where the DB must be the unambiguous source of truth; pick event sourcing (Kafka-first) for ML feature stores where seconds of staleness is acceptable — this is the explicit recommendation in the file; pick Saga specifically for multi-step distributed transactions without an available transaction coordinator.
- Key operational detail: "for ML feature stores, event sourcing (Kafka-first) is the pragmatic default" per the file's own interview-answer guidance.

**Circuit Breaker + Fallback Hierarchy**
- What it is: a 5-level fallback chain (primary online features+model → cached prediction → simpler in-process fallback model → hard-coded business rules → default action) combined with a circuit breaker (CLOSED/OPEN/HALF_OPEN) that stops calling a failing dependency and serves a fallback instead.
- Pros: graceful degradation instead of total outage; circuit breaker prevents cascading failure/timeout pile-up.
- Cons: each fallback level is a lower-quality prediction — must decide acceptable risk tolerance for "default allow" vs "default block" at the last resort.
- Pick over alternatives: pick a full fallback hierarchy over a simple retry-or-fail approach whenever the serving path has hard latency SLAs (e.g., <100ms) and dependencies (feature store, model server) can degrade independently.
- Key operational detail: feature-store timeout should be ~10% of total latency budget (e.g., 10ms timeout within a 100ms SLA) so a hanging call doesn't consume the whole budget before falling back.

---

## Quick Cross-Reference: Where Each Topic Lives

| Topic | Primary file |
|---|---|
| Deployment strategies, latency optimization | `02-deployment-patterns.md` |
| MLOps lifecycle, CI/CD, cloud platforms | `01-mlops.md` |
| Registry metadata, approval gates, compliance | `03-model-governance.md` |
| 20 ML design patterns | `system-design/02-machine-learning-design-patterns.md` |
| Feature store deep-dive | `system-design/08-feature-store-architecture.md` |
| Registry tooling + canary/rollback mechanics | `system-design/10-model-registry-versioning.md` |
| A/B testing, bandits, interleaving | `system-design/14-ab-testing-experimentation.md` |
| Real-time systems, consistency, fallback | `system-design/05-real-time-ml-systems.md` |
| LLM serving internals (KV cache, batching) | `system-design/12-llm-inference-ops.md` |
| Cost optimization | `system-design/13-cost-optimization.md` |
| Full MLE lifecycle (data → monitoring) | `system-design/01-machine-learning-engineering.md` |
| 10-minute condensed revision | `REVISION.md` |
