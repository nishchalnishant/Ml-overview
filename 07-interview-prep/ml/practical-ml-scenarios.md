---
module: Interview Prep
topic: Ml
subtopic: Practical Ml Scenarios
status: unread
tags: [interviewprep, ml, ml-practical-ml-scenarios]
---
# Practical ML Scenarios

## What This File Is For

Scenario questions test whether you can decompose a production failure, identify the root cause, and prescribe a fix — without retraining immediately as a reflex. The structure for each scenario:

1. What the interviewer is actually testing — the underlying competency
2. The reasoning structure — why first-principles thinkers approach it this way
3. The pattern in action — a worked example
4. Common traps — where people go wrong and why

---

# 1. Model Great in Training, Bad in Production

## What the interviewer is actually testing

Whether you approach production failures as systems problems rather than model problems. Retraining is not the first move — it is what you do after you have excluded simpler explanations.

## The reasoning structure

When a model performs well offline and fails in production, there are five distinct failure modes, each requiring a different fix:

**1. Data leakage during training.** A feature uses information that would not be available at prediction time. The model learned a shortcut that does not generalize.

**2. Train-serve skew.** The feature computation logic in training differs from the logic in production. The model sees different distributions at serving time than it was trained on.

**3. Distribution shift.** The production data distribution has changed since training (data drift) or the underlying relationship between features and labels has changed (concept drift).

**4. Unrealistic validation.** The offline evaluation setup did not mirror production. Random splits on time-series data, evaluation on a non-representative holdout, or evaluation at the wrong temporal granularity.

**5. Threshold mismatch.** The classification threshold was optimized for offline data but the production score distribution differs.

**Diagnostic order:**

1. Compare feature distributions: compute PSI on key features between training data and recent production data. Spike in PSI → train-serve skew or drift.
2. Compare prediction distributions: look at score distributions in production vs. at evaluation time. If scores are compressed or shifted, the model is receiving different inputs.
3. Check feature computation: diff the training preprocessing code against the serving preprocessing code. Find any inconsistency.
4. Check for leakage: for each feature, verify it was available at the prediction timestamp. Point-in-time correctness.
5. Only after these: consider retraining.

## The pattern in action

A churn prediction model has AUC 0.84 offline. After deployment, the product team reports interventions are failing at much higher rates than expected.

Investigation:
- PSI on "days since last login": 0.31 (above the 0.25 threshold). The offline training data was from a period of normal usage. A product change two weeks ago changed login patterns significantly.
- Prediction distribution: scores cluster near 0.2–0.4. Offline they were distributed across 0.1–0.9. The model is not discriminating.
- Feature computation: offline "days since login" was capped at 30 days. Serving version was not capped and includes users with 180+ days — an entire bucket that training never saw.

Fix: the serving-side feature computation bug (missing cap) is the root cause. Fix the cap, and the prediction distribution recovers. Drift is a secondary issue — address with a scheduled retrain, but only after the feature bug is fixed.

## Common traps

**Retraining before diagnosing.** Retraining on production data that has the same train-serve skew will not fix the problem. The new model will exhibit the same behavior.

**"It must be drift."** Drift is the last thing to blame, not the first. Drift cannot be fixed by debugging; it requires data collection and retraining. Skew and leakage can be fixed by correcting code. Check for code bugs first.

**Only checking aggregate metrics.** AUC can look stable while a specific user segment degrades. Disaggregate the production metrics by feature bucket, demographic, or acquisition cohort to isolate where the failure is concentrated.

---

# 2. Extreme Class Imbalance

## What the interviewer is actually testing

Whether you understand that accuracy is a meaningless metric for highly imbalanced problems, and whether you can reason about the cost structure of false positives versus false negatives in a specific domain.

## The reasoning structure

**Why accuracy fails.** Fraud at 0.01% prevalence: predicting "not fraud" for every transaction gives 99.99% accuracy and zero utility. Accuracy optimizes for the majority class and reveals nothing about the minority class performance.

**Start from the cost structure.** In fraud detection:
- False negative (missed fraud): financial loss plus reputational damage
- False positive (flagged legitimate transaction): review cost plus customer friction

The business values these differently. The operating threshold should be set to minimize total expected cost, not to maximize accuracy or even F1.

**Right metrics:**
- Precision-recall AUC: measures the tradeoff between catching fraud and generating false alarms, independent of class balance
- Recall at target precision: what fraction of fraud do we catch if we constrain false positive rate to X%?
- Cost-sensitive evaluation: expected cost = FN_cost × FN_rate + FP_cost × FP_rate

**Approaches to handling imbalance (in order of preference):**

1. **Adjust threshold.** The model's probability output is not calibrated to 0.5 as the decision boundary. Set the threshold based on the precision-recall tradeoff that matches the cost structure. Cost: zero.

2. **Class weighting.** `class_weight = {0: 1, 1: 10000}` in the loss function. Effectively upsamples the minority class during gradient computation. Cost: zero implementation overhead.

3. **Focal loss.** Downweights easy negatives (clearly legitimate transactions). Forces the model to focus on hard boundary cases. `FL(p_t) = -(1-p_t)^γ log(p_t)`. Effective for extremely imbalanced detection tasks.

4. **Undersampling / oversampling.** Undersample majority class or oversample minority class (SMOTE). Useful when class weighting is insufficient, but SMOTE can generate unrealistic samples in high dimensions.

## The pattern in action

**Fraud at 0.01% base rate.** Training set: 10M transactions, 1,000 fraud cases.

Model produces probability scores. Threshold at 0.5: precision = 0.92, recall = 0.18. Most fraud is missed. Business impact: 82% of fraud goes uncaught.

Threshold optimization: plot precision-recall curve. Find the threshold where recall = 0.70 at acceptable precision (say, 0.30). This means a human review queue catches 70% of fraud with a 70% false alarm rate — which may be acceptable given the cost structure.

Setting `class_weight='balanced'`: model assigns weight 10,000 to each fraud example during training. After retraining with adjusted threshold: precision = 0.28, recall = 0.73. Review queue volume is manageable.

Adding focal loss (γ=2): further focuses training on the hardest fraud cases. Recall improves to 0.79 at precision = 0.31.

Final evaluation metric: cost per transaction = 0.79 × (fraud_rate × fraud_loss) avoided − review_cost_per_alert × alert_rate. Not AUC.

## Common traps

**"We got 99.9% accuracy."** Never use accuracy on imbalanced problems. Report precision, recall, F1 at the operating threshold, or precision-recall AUC.

**Using SMOTE before checking class weights.** Class weights are equivalent to oversampling and have no failure modes. SMOTE can generate unrealistic samples. Try class weights first.

**Setting the threshold at 0.5.** For rare events, the probability of any transaction being fraud is much lower than 0.5 even for true fraud cases. The threshold should be set based on the precision-recall tradeoff, not assumed to be 0.5.

---

# 3. RAG System Is Hallucinating

## What the interviewer is actually testing

Whether you can diagnose which component of the RAG pipeline is failing — retrieval or generation — and apply targeted fixes rather than treating hallucination as an LLM problem.

## The reasoning structure

**RAG has two failure modes:**

**1. Retrieval failure.** The relevant context was not retrieved. The LLM was asked to answer without the information it needed, so it generated a plausible-sounding but fabricated answer.

**2. Generation failure.** The relevant context was retrieved but the LLM ignored it, summarized it incorrectly, or confabulated beyond what the context supports.

**Diagnosis first:** add logging to the RAG pipeline. For each hallucinated response:
- Was the correct source document in the top-k retrieved context? (Retrieval recall)
- If yes, did the answer contradict the retrieved context? (Generation faithfulness)
- If the source was retrieved, was it at position 1 or position 8? (Position matters — "lost in the middle" effect)

**Retrieval-side fixes:**
- Chunking strategy: if chunks are too large, the relevant information is diluted. If too small, context is fragmented. Typical sweet spot: 300–500 tokens with 50-token overlap.
- Retrieval method: BM25 (keyword matching) for precise factual queries; dense retrieval (vector similarity) for semantic queries. Hybrid often outperforms either alone.
- Reranking: a cross-encoder reranker (e.g., a BERT-based relevance model) reranks top-k candidates with much higher precision than vector similarity.
- Metadata filtering: filter retrieved chunks by document type, date, source before final selection.

**Generation-side fixes:**
- Prompt constraints: "Answer only based on the provided context. If the information is not in the context, say 'I don't know.'" This reduces hallucination but may increase non-answers.
- Grounded citation: require the model to cite which retrieved passage supports each claim. If it cannot cite, it should not claim.
- Answer verification: a separate model or rule checks whether each sentence in the response is supported by any retrieved passage (Natural Language Inference or string overlap).
- Temperature: lower temperature reduces creative generation but also reduces hallucination.

## The pattern in action

**Enterprise Q&A system.** Users ask questions about internal policies. Hallucination rate: 23% of answers contain at least one factually unsupported claim.

Diagnosis:
- Retrieval recall@5 (fraction of queries where the answer-containing chunk is in the top 5): 0.61. This means 39% of queries fail at retrieval.
- Of the 61% where retrieval succeeds: generation faithfulness (answer consistent with retrieved context) = 0.81. So 19% of successful retrievals still hallucinate.

Two-stage fix:
1. Improve retrieval: switch from 1000-token chunks to 400-token chunks with 50-token overlap. Add a BM25 first-stage retrieval before dense reranking. Retrieval recall@5 improves to 0.84.
2. Improve generation: add a citation requirement to the prompt. Add an NLI-based faithfulness checker as a post-processing step. Flag responses where the checker confidence is low for human review rather than returning them to the user.

Result: hallucination rate drops from 23% to 5% on the evaluation set.

## Common traps

**Treating all hallucination as a model quality problem.** Most hallucination in RAG systems is retrieval failure. Improving the LLM does not help if the relevant context is not retrieved.

**Not evaluating retrieval separately from generation.** You cannot diagnose the system without measuring retrieval recall and generation faithfulness independently. Add logging for both.

**Over-constraining prompts.** A prompt that says "only answer from context" reduces hallucination but increases the rate of "I don't know" responses on questions that the context does partially address. Tune the constraint to match user tolerance.

---

# 4. Real-Time Recommendation System Has High Latency

## What the interviewer is actually testing

Whether you approach latency as a systems engineering problem — identifying where time is spent before optimizing — rather than immediately jumping to model compression.

## The reasoning structure

**Profile before optimizing.** P99 latency = 850ms, budget = 200ms. Where does the time go? This must be measured, not assumed.

**The latency budget in a typical recommendation pipeline:**

| Stage | Typical range |
|-------|---------------|
| API gateway overhead | 5–15ms |
| Feature retrieval (online store lookup) | 20–80ms |
| Model forward pass (CPU) | 50–500ms |
| Model forward pass (GPU) | 5–50ms |
| Post-processing / business rules | 5–20ms |
| Serialization and network | 10–30ms |

If feature retrieval is 300ms and the model is 50ms, optimizing the model gives no improvement. The bottleneck must be profiled.

**Optimization approaches, by root cause:**

**Model is the bottleneck:**
- Quantization: int8 inference reduces memory bandwidth pressure. 2–4× speedup on CPU, 1.5–2× on GPU.
- Distillation: train a smaller student model (4-layer BERT instead of 12-layer). 3–5× speedup with 2–3% quality loss.
- Pruning: remove attention heads or MLP neurons with low gradient norms. Structured pruning gives real speedups; unstructured pruning does not.
- ONNX + TensorRT compilation: graph optimization and kernel fusion. 1.5–3× improvement without model changes.

**Feature retrieval is the bottleneck:**
- Precompute user and item embeddings in batch (hourly or daily). Online lookup is O(1) instead of O(feature computation).
- Use in-process caching: cache recent lookups to avoid repeated Redis calls for the same user within a session.

**Model should not be on the critical path:**
- Two-stage architecture: fast ANN retrieval (10–20ms) to get 500 candidates, then a lighter ranker on candidates. The expensive model does not score all items.
- Asynchronous precomputation: compute recommendations for users who are likely to visit soon (e.g., users active in the last hour) in background, serve from cache.

## The pattern in action

**Profiling reveals:** feature retrieval (Redis lookups for user embedding + 50 item embeddings) = 420ms. Model forward pass = 120ms. Everything else = 30ms.

The bottleneck is feature retrieval, not the model.

Fix: user embedding is static (updated hourly). Move it to local memory cache per serving pod. Instead of 50 individual Redis lookups for item embeddings, batch them into one pipelined Redis call. Feature retrieval drops from 420ms to 45ms.

Now model forward pass (120ms) is the bottleneck. Apply int8 quantization: 120ms → 60ms. Total: 45 + 60 + 30 = 135ms. Within 200ms budget.

## Common traps

**Optimizing the model without profiling.** Distillation takes weeks of engineering. If the bottleneck is a Redis lookup, distillation wastes time.

**Adding more hardware as the first solution.** Scaling horizontally reduces throughput bottlenecks, not latency bottlenecks. If the P99 latency is caused by a 400ms Redis call, adding more pods does not help.

**Forgetting about warm-up time.** Neural network inference has a warm-up period on the first request (JIT compilation, GPU initialization). P99 latency measured without warm-up is misleading. Profile with warmed-up systems.

---

# 5. Cold Start in Recommendation

## What the interviewer is actually testing

Whether you understand the cold start problem as a design constraint that requires explicit architectural decisions, not just a limitation to apologize for.

## The reasoning structure

**Cold start has three forms:**

**User cold start:** new user, no interaction history. Collaborative filtering has nothing to compute.

**Item cold start:** new item, no interactions. The item cannot appear in CF recommendations until it accumulates data.

**System cold start:** no interactions at all. New product launch where the interaction graph is empty.

**Each requires a different solution:**

**User cold start → fallback to content-based:**
- Onboarding quiz: ask the user to rate 5–10 items or select preferred categories. Build a content-based user profile immediately.
- Demographic priors: use coarse user attributes (location, device type) to assign a population-level prior. "New users from this region typically engage with X."
- Popularity: show globally popular items while personalization data accumulates. Not ideal, but better than random.

**Item cold start → use item attributes:**
- Content-based similarity: embed new items using their metadata (genre, tags, description, cast). Find existing items with similar embeddings. Recommend to users who interacted with those similar items.
- Cold start item injection: mix a fraction of new items into all users' recommendations. Collect feedback to bootstrap the CF signal.
- Context-aware popularity: a new movie in genre X should be shown to users with strong genre X preference, not to the general population.

**System cold start → collect data deliberately:**
- Run a content-based system initially
- Track all user interactions with full logging
- Switch to collaborative filtering after enough signal accumulates
- Define the transition threshold: "when the CF model's offline AUC on held-out interactions exceeds content-based by X%, switch"

## The pattern in action

**Streaming service launch.** Day 1: 100,000 new users, 1,000 titles, 0 interactions.

Day 1–7: content-based system. User selects 3 genres and 5 previously seen titles during onboarding. Recommend titles similar to selected content.

Day 7–30: hybrid system. CF signal is thin but exists. Weighted combination: 20% CF + 80% content-based. New titles injected at 10% of each user's recommendation list.

Day 30+: CF dominates as interaction matrix densifies. New user cold start: 100% content-based for the first session, transition to hybrid after 5 interactions.

**Item cold start for new releases:** before a new title launches, embed it using metadata (director, genre, keywords, similar title list from editorial team). On launch day, the title can immediately appear for users whose CF profile matches the metadata embedding cluster.

## Common traps

**Treating cold start as a temporary problem.** On any large platform, 20–30% of users are new at any given time. Cold start is a permanent system requirement, not a bootstrapping phase.

**Not measuring cold start performance separately.** Overall recommendation quality metrics (NDCG on all users) obscure cold start performance. Measure new-user NDCG and new-item coverage separately.

**Onboarding questions that are too long.** Users abandon onboarding if it takes more than 2–3 minutes. Asking for ratings on 50 titles collects better data but loses 80% of users. Design onboarding around what users will actually complete.

---

# 6. Concept Drift

## What the interviewer is actually testing

Whether you can distinguish the cause of model degradation (the underlying relationship changed, not just the inputs) and design a monitoring and response system accordingly.

## The reasoning structure

**Concept drift vs. data drift:**
- Data drift: P(X) changes. The input distribution shifts. The model may still perform well if P(Y|X) is stable.
- Concept drift: P(Y|X) changes. The same input now has a different label. The model's learned function is no longer correct regardless of how the inputs are distributed.

**Why concept drift is harder than data drift:**
- Data drift can be detected by monitoring input distributions (PSI, KS test). No labels required.
- Concept drift can only be detected by monitoring prediction accuracy. This requires ground truth labels, which often arrive with a delay.

**Detecting concept drift without waiting for labels:**
- Monitor prediction distributions: if the model's score distribution shifts substantially while input distributions remain stable, the model is responding differently to the same inputs.
- Proxy signals: in fraud, use chargeback rates as a lagged label. In recommendations, use downstream engagement as a label proxy.
- Drift detectors: ADWIN (Adaptive Windowing) or DDM (Drift Detection Method) on incoming predictions or calibrated errors.

**Response to concept drift:**
1. Confirm it is concept drift, not data drift or train-serve skew
2. Identify which part of the feature-label relationship has changed (which feature's predictive value has degraded?)
3. Collect labeled examples from the new distribution
4. Retrain on a window of recent data (or a weighted combination of recent and historical data)
5. Compare old model vs. new model on held-out recent data before deploying

## The pattern in action

**Fraud model.** Base rate was stable at 0.5% fraud for 18 months. A new fraud scheme emerges that targets a different payment channel with different behavioral signatures.

Detection: the fraud model's precision on confirmed fraud cases (labeled by the fraud operations team) drops from 0.72 to 0.41 over 3 weeks. The input distribution (PSI on key features) is stable — transactions look the same from the feature perspective. But the fraud-legitimate relationship has changed.

Response:
1. Tag the new fraud scheme in the operations team's review queue
2. Collect 500 confirmed examples of the new scheme
3. Retrain with a recency-weighted training window (last 6 months weighted 3×, older data weighted 1×)
4. The new scheme has features (transaction channel, recipient type) that were informative in the new regime but not in the old one — include these as new features
5. A/B test old vs. new model specifically on transactions of the new scheme type

## Common traps

**Retraining on all historical data equally.** For concept drift, older data may actively hurt performance if the old P(Y|X) differs from the new one. Use recency weighting or a sliding window.

**Detecting concept drift from prediction distributions alone.** Prediction distribution shifts can be caused by data drift (new type of transaction), model degradation (upstream feature bug), or concept drift. Ruling out the first two before concluding concept drift avoids unnecessary retraining.

**Conflating seasonal patterns with concept drift.** A fraud model that degrades every November is not experiencing concept drift — it is experiencing seasonal distribution shift that is predictable and should be handled with seasonality-aware training, not ad-hoc retraining.

---

# 7. Feedback Loops

## What the interviewer is actually testing

Whether you recognize that a deployed model changes the distribution of future training data, and whether you can reason about the long-term consequences of this self-reinforcing cycle.

## The reasoning structure

**The mechanism.** A model deployed in production influences user behavior or business actions. Those actions generate new data. That data is used to train the next model. The model's decisions become part of its own training signal.

**Types of feedback loops:**

**Popularity bias in recommendations.** The recommendation model surfaces popular items more often. Popular items get more clicks. More clicks generate more training signal for popular items. The next model recommends popular items even more. Long-tail items starve for signal and eventually disappear from recommendations.

**Predictive policing.** A model predicts high-crime probability in specific neighborhoods. Officers are sent there. More arrests are made there. The next model trains on data that is denser in those neighborhoods. The model predicts higher crime in the same neighborhoods.

**Moderation model skew.** A content moderation model labels borderline content as policy-violating. Users stop posting borderline content of that type. The next model trains on data without that borderline type. The model's decision boundary shifts.

**Why feedback loops are hard:**

The training data becomes a function of past model decisions. The i.i.d. assumption is violated. The true data distribution (without model influence) is not observable. The model's apparent performance can improve while its behavior drifts in ways that harm users or upstream metrics.

**Mitigation:**

1. **Exploration:** deliberately show a fraction of random or diverse recommendations (ε-greedy or Thompson sampling). This prevents the positive feedback loop on popular items and maintains signal for long-tail items.

2. **Counterfactual evaluation:** log what the model would have shown for each user alongside what it actually showed. Measure outcomes under both. This is inverse propensity scoring (IPS) — estimate what the click rate would have been if you had shown a different item.

3. **Corrective upweighting:** in training, upweight examples that were not selected by the model (under-explored) to counteract selection bias.

4. **Separate exploration from exploitation:** serve a dedicated exploration model to a fixed fraction of users. Use that data for unbiased training signal. Use the main model for the rest.

## The pattern in action

**News recommendation.** Feedback loop: the model learns that outrage-inducing headlines get more clicks. It surfaces those articles more. Users click more. Training data concentrates outrage signal. The model amplifies outrage.

Detection: track content diversity over time. Measure the "topic concentration" metric (entropy of topics in the recommendation distribution). If topic entropy is declining, the feedback loop is narrowing recommendations.

Mitigation: add a diversity constraint to the ranker. Penalize recommending more than 2 items from the same category to the same user in one session. Separately, add a 5% exploration allocation — random high-quality items from underrepresented categories — to maintain signal on content that the model is not currently surfacing.

## Common traps

**Not logging counterfactuals.** Once the model is deployed, you can only observe outcomes for the items it showed. Items it did not show get zero signal. Without counterfactual logging, you cannot train an unbiased next model. This logging is cheap to add at deployment time and very expensive to reconstruct after the fact.

**Ignoring feedback loops until the harm is visible.** By the time topic diversity has collapsed or a demographic group is being systematically over-policed, the loop has been running for a long time. Monitor diversity and representation metrics from day 1.

---

# 8. Vanishing and Exploding Gradients

## What the interviewer is actually testing

Whether you can trace gradient flow through a network, identify where it breaks down, and explain why architectural choices like residual connections and normalization are solutions, not just design preferences.

## The reasoning structure

**Vanishing gradients.** During backpropagation, the gradient of the loss with respect to early layers is computed via repeated chain rule multiplication. If each layer's gradient factor is < 1 (as with sigmoid: σ'(z) ≤ 0.25 maximum), gradients shrink exponentially with depth.

For L layers with sigmoid activations: gradient ≈ 0.25^L. For L=20: gradient ≈ 10^{-12}. Early layers receive essentially zero gradient and do not learn.

**Exploding gradients.** If gradient factors are > 1 (e.g., poorly initialized weights with large variance), gradients grow exponentially. The optimizer takes steps that are too large, and training diverges.

**How each architectural solution addresses the problem:**

**ReLU activations.** ReLU'(z) = 1 for z > 0, 0 for z < 0. For active neurons, the gradient passes through unchanged (no shrinking). For dead neurons (z ≤ 0), the gradient is 0. Eliminates the 0.25 factor from sigmoid.

**Residual connections (He et al., 2016).** `y = F(x) + x`. The gradient of the loss with respect to x is: `∂L/∂x = ∂L/∂y · (1 + ∂F/∂x)`. The "+1" term provides an identity gradient path from any layer directly to the input. Even if ∂F/∂x vanishes, ∂L/∂x ≈ ∂L/∂y (the upstream gradient passes through unchanged).

**Normalization (BatchNorm, LayerNorm).** Controls the scale of activations at each layer. Prevents the internal covariate shift that can cause gradients to saturate.

**Gradient clipping.** Clips the global gradient norm to a maximum value before the optimizer step. Prevents exploding gradients from causing divergence. `g = g × min(1, clip_value / ‖g‖)`.

**LSTM/GRU for RNNs.** The standard RNN gradient through T time steps = product of T Jacobians. LSTM's gated architecture provides a cell state with additive updates (not multiplicative) — analogous to residual connections for time.

## The pattern in action

**Diagnosing vanishing gradients in a 50-layer MLP without residuals:**

Symptom: training loss decreases slowly. Plotting gradient norms per layer shows: layer 1 gradient norm = 10^{-8}, layer 50 gradient norm = 0.1. The first layers are not learning.

Fix: add residual connections every 2 layers. After fix: layer 1 gradient norm = 0.08. Training converges 5× faster.

**Diagnosing exploding gradients in an RNN:**

Symptom: training loss oscillates wildly, then hits NaN. Gradient norm spikes to 10^6 at step 1500.

Fix: gradient clipping at norm = 5.0. Training stabilizes immediately.

## Common traps

**Using sigmoid/tanh for deep networks.** Sigmoid's maximum gradient of 0.25 means depth 20+ networks cannot train without residuals or careful initialization. ReLU is the default for deep networks.

**Not initializing weights carefully.** Xavier/Glorot initialization keeps gradient variance stable across layers for tanh. He initialization is appropriate for ReLU. Using default random initialization in a deep network can cause exploding gradients at initialization.

---

# 9. Multicollinearity

## What the interviewer is actually testing

Whether you understand the difference between multicollinearity's impact on prediction vs. interpretation, and whether you can select the right fix for the right goal.

## The reasoning structure

**What multicollinearity is.** Two or more features are highly correlated. In linear regression: `β = (X^T X)^{-1} X^T y`. If X^T X is near-singular (because columns are nearly linearly dependent), the inverse is unstable. Small changes in data can produce large changes in the estimated coefficients.

**The key distinction:**

Multicollinearity affects **coefficient interpretation** more than **prediction accuracy**.

If features A and B are perfectly correlated, the model cannot distinguish how much of the prediction should be attributed to A vs. B. But the combined prediction (A's contribution + B's contribution) is still accurate.

For prediction: multicollinearity is often acceptable. The model will still predict well on new data from the same distribution.

For interpretation: multicollinearity makes coefficients untrustworthy. A large positive coefficient on A and a large negative on B, where A≈B, means both coefficients are artifacts of numerical instability.

**When it matters most:**
- Interpreting coefficients (regulatory requirements, fairness analysis)
- Stable coefficient estimates across different data samples
- Feature selection based on coefficient magnitude

**Fixes:**

1. **Drop redundant features.** If A and B have correlation 0.97, remove one. Use VIF (Variance Inflation Factor) to identify culprits. VIF > 10 typically warrants investigation.

2. **Ridge regularization.** Adds L2 penalty to the loss: `min ‖Xβ - y‖² + λ‖β‖²`. Ridge shrinks correlated features toward each other and stabilizes the solution. Does not zero out features, but reduces instability.

3. **PCA.** Transform correlated features into orthogonal principal components. Train on the transformed features. Eliminates collinearity by construction. Loss: interpretability in the original feature space.

4. **Partial least squares (PLS).** Like PCA but finds components that maximally predict y (not just maximally explain X variance). Better than PCA when the correlated features are all predictive.

## The pattern in action

**House price prediction.** Features include: `square_footage`, `number_of_rooms`, `floor_area`. These three are highly correlated (VIF > 15 for all).

If the goal is prediction: use ridge regularization. Coefficient instability does not matter; prediction on new houses is accurate.

If the goal is "how much does each square foot contribute to price?" (regulatory pricing audit): drop `number_of_rooms` and `floor_area`, keep `square_footage`. Run a separate analysis of whether number of rooms contributes beyond square footage (partial correlation).

## Common traps

**Treating multicollinearity as always harmful.** For prediction problems without interpretability requirements, correlated features often do not hurt and can help (they provide redundancy). Only fix multicollinearity when you need stable coefficients or feature interpretations.

**Removing features without checking VIF after removal.** Removing one feature from a highly correlated group may not solve the problem if three other correlated features remain. Recompute VIF after each removal.

---

# 10. Scenario Rapid-Fire

## Small object detection is poor

**Diagnosis order:**
1. Input resolution: downsampled images lose small objects entirely. Increase input resolution.
2. Feature pyramid network (FPN): single-scale features cannot represent small objects. FPN creates multi-scale feature maps.
3. Anchor design: if smallest anchor size is 32×32px and target objects are 8×8px, no anchor matches. Reduce minimum anchor size or switch to anchor-free detection.
4. Label quality: verify that small objects are labeled in the training data. Objects < 10px are often skipped in annotation.

## Fraud model catches too little (low recall)

**Diagnosis order:**
1. Threshold: is the threshold set too conservatively? Lower it and measure precision cost.
2. Class weighting: is the minority class weighted sufficiently during training?
3. Label delay: confirmed fraud labels arrive with a delay. Are you training on too many unlabeled (uncertain) examples?
4. Feature coverage: do recent fraud patterns have features in the model? A new fraud scheme may not be captured by existing features.

## Recommendation system feels repetitive

**Diagnosis order:**
1. Diversity metrics: measure topic concentration in recommendation lists. Is entropy declining?
2. Popularity bias: are the top-10 recommended items the same popular items for most users?
3. Exploration: is there an exploration budget (e.g., 10% random items)?
4. Feedback loop: have recommendations been narrowing over time as training data concentrates?

## Time-series model looks amazing offline but fails in production

**Diagnosis order:**
1. Split strategy: was the validation split temporal? Random splits on time-series data produce unrealistically high offline metrics.
2. Leakage: do any features use future data (future averages, forward-filled values)?
3. Forecast horizon: offline evaluation on 1-step-ahead but production needs 7-step-ahead forecasts.
4. Data freshness: are the most recent data points used in production actually available during training?

---

# Quick Diagnostics

**When a production model degrades:**

1. Check feature distributions (PSI) — rules out train-serve skew
2. Check prediction distributions — identifies scoring anomalies
3. Check for upstream data changes (schema, library updates) — rules out infrastructure-level skew
4. Only after the above: consider drift, then retrain

The diagnostic order matters. Retraining before diagnosing the root cause fixes nothing if the failure is a feature computation bug.

**The scenario answer formula:**

1. Name the likely failure modes (typically 3–5 candidates)
2. State what to inspect first and why (prioritize by evidence cost and likelihood)
3. Explain how you would validate the diagnosis
4. Describe the fix that addresses the root cause
5. Add monitoring to detect the same failure earlier next time

## Rapid Recall

### PSI on "days since last login"
- Direct Answer: 0.31 (above the 0.25 threshold). The offline training data was from a period of normal usage. A product change two weeks ago changed login patterns significantly.
- Why: This matters because it tells you how to reason about psi on "days since last login".
- Pitfall: Don't answer "PSI on "days since last login"" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: 0.31 (above the 0.25 threshold). The offline training data was from a period of normal usage. A product change two weeks ago changed login patterns significantly.

### Prediction distribution
- Direct Answer: scores cluster near 0.2–0.4. Offline they were distributed across 0.1–0.9. The model is not discriminating.
- Why: This matters because it tells you how to reason about prediction distribution.
- Pitfall: Don't answer "Prediction distribution" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: scores cluster near 0.2–0.4. Offline they were distributed across 0.1–0.9. The model is not discriminating.

### Feature computation: offline "days since login" was capped at 30 days. Serving version was not capped and includes users with 180+ days
- Direct Answer: an entire bucket that training never saw.
- Why: This matters because it tells you how to reason about feature computation: offline "days since login" was capped at 30 days. serving version was not capped and includes users with 180+ days.
- Pitfall: Don't answer "Feature computation: offline "days since login" was capped at 30 days. Serving version was not capped and includes users with 180+ days" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: an entire bucket that training never saw.

### False negative (missed fraud)
- Direct Answer: financial loss plus reputational damage
- Why: This matters because it tells you how to reason about false negative (missed fraud).
- Pitfall: Don't answer "False negative (missed fraud)" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: financial loss plus reputational damage

### False positive (flagged legitimate transaction)
- Direct Answer: review cost plus customer friction
- Why: This matters because it tells you how to reason about false positive (flagged legitimate transaction).
- Pitfall: Don't answer "False positive (flagged legitimate transaction)" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: review cost plus customer friction

### Precision-recall AUC
- Direct Answer: measures the tradeoff between catching fraud and generating false alarms, independent of class balance
- Why: This matters because it tells you how to reason about precision-recall auc.
- Pitfall: Don't answer "Precision-recall AUC" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: measures the tradeoff between catching fraud and generating false alarms, independent of class balance

### Recall at target precision
- Direct Answer: what fraction of fraud do we catch if we constrain false positive rate to X%?
- Why: This matters because it tells you how to reason about recall at target precision.
- Pitfall: Don't answer "Recall at target precision" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: what fraction of fraud do we catch if we constrain false positive rate to X%?

### Cost-sensitive evaluation
- Direct Answer: expected cost = FN_cost × FN_rate + FP_cost × FP_rate
- Why: This matters because it tells you how to reason about cost-sensitive evaluation.
- Pitfall: Don't answer "Cost-sensitive evaluation" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: expected cost = FN_cost × FN_rate + FP_cost × FP_rate

### Was the correct source document in the top-k retrieved context? (Retrieval recall)
- Direct Answer: Was the correct source document in the top-k retrieved context? (Retrieval recall)
- Why: This matters because it tells you how to reason about was the correct source document in the top-k retrieved context? (retrieval recall).
- Pitfall: Don't answer "Was the correct source document in the top-k retrieved context? (Retrieval recall)" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Was the correct source document in the top-k retrieved context? (Retrieval recall)

### If yes, did the answer contradict the retrieved context? (Generation faithfulness)
- Direct Answer: If yes, did the answer contradict the retrieved context? (Generation faithfulness)
- Why: This matters because it tells you how to reason about if yes, did the answer contradict the retrieved context? (generation faithfulness).
- Pitfall: Don't answer "If yes, did the answer contradict the retrieved context? (Generation faithfulness)" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: If yes, did the answer contradict the retrieved context? (Generation faithfulness)

### If the source was retrieved, was it at position 1 or position 8? (Position matters
- Direct Answer: "lost in the middle" effect)
- Why: This matters because it tells you how to reason about if the source was retrieved, was it at position 1 or position 8? (position matters.
- Pitfall: Don't answer "If the source was retrieved, was it at position 1 or position 8? (Position matters" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: "lost in the middle" effect)

### Chunking strategy
- Direct Answer: if chunks are too large, the relevant information is diluted. If too small, context is fragmented. Typical sweet spot: 300–500 tokens with 50-token overlap.
- Why: This matters because it tells you how to reason about chunking strategy.
- Pitfall: Don't answer "Chunking strategy" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: if chunks are too large, the relevant information is diluted. If too small, context is fragmented. Typical sweet spot: 300–500 tokens with 50-token overlap.

### Retrieval method
- Direct Answer: BM25 (keyword matching) for precise factual queries; dense retrieval (vector similarity) for semantic queries. Hybrid often outperforms either alone.
- Why: This matters because it tells you how to reason about retrieval method.
- Pitfall: Don't answer "Retrieval method" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: BM25 (keyword matching) for precise factual queries; dense retrieval (vector similarity) for semantic queries. Hybrid often outperforms either alone.

### Reranking
- Direct Answer: a cross-encoder reranker (e.g., a BERT-based relevance model) reranks top-k candidates with much higher precision than vector similarity.
- Why: This matters because it tells you how to reason about reranking.
- Pitfall: Don't answer "Reranking" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: a cross-encoder reranker (e.g., a BERT-based relevance model) reranks top-k candidates with much higher precision than vector similarity.

### Metadata filtering
- Direct Answer: filter retrieved chunks by document type, date, source before final selection.
- Why: This matters because it tells you how to reason about metadata filtering.
- Pitfall: Don't answer "Metadata filtering" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: filter retrieved chunks by document type, date, source before final selection.

### Prompt constraints
- Direct Answer: "Answer only based on the provided context. If the information is not in the context, say 'I don't know.'" This reduces hallucination but may increase non-answers.
- Why: This matters because it tells you how to reason about prompt constraints.
- Pitfall: Don't answer "Prompt constraints" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: "Answer only based on the provided context. If the information is not in the context, say 'I don't know.'" This reduces hallucination but may increase non-answers.

### Grounded citation
- Direct Answer: require the model to cite which retrieved passage supports each claim. If it cannot cite, it should not claim.
- Why: This matters because it tells you how to reason about grounded citation.
- Pitfall: Don't answer "Grounded citation" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: require the model to cite which retrieved passage supports each claim. If it cannot cite, it should not claim.

### Answer verification
- Direct Answer: a separate model or rule checks whether each sentence in the response is supported by any retrieved passage (Natural Language Inference or string overlap).
- Why: This matters because it tells you how to reason about answer verification.
- Pitfall: Don't answer "Answer verification" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: a separate model or rule checks whether each sentence in the response is supported by any retrieved passage (Natural Language Inference or string overlap).

### Temperature
- Direct Answer: lower temperature reduces creative generation but also reduces hallucination.
- Why: This matters because it tells you how to reason about temperature.
- Pitfall: Don't answer "Temperature" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: lower temperature reduces creative generation but also reduces hallucination.

### Retrieval recall@5 (fraction of queries where the answer-containing chunk is in the top 5)
- Direct Answer: 0.61. This means 39% of queries fail at retrieval.
- Why: This matters because it tells you how to reason about retrieval recall@5 (fraction of queries where the answer-containing chunk is in the top 5).
- Pitfall: Don't answer "Retrieval recall@5 (fraction of queries where the answer-containing chunk is in the top 5)" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: 0.61. This means 39% of queries fail at retrieval.

### Of the 61% where retrieval succeeds
- Direct Answer: generation faithfulness (answer consistent with retrieved context) = 0.81. So 19% of successful retrievals still hallucinate.
- Why: This matters because it tells you how to reason about of the 61% where retrieval succeeds.
- Pitfall: Don't answer "Of the 61% where retrieval succeeds" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: generation faithfulness (answer consistent with retrieved context) = 0.81. So 19% of successful retrievals still hallucinate.

### Quantization
- Direct Answer: int8 inference reduces memory bandwidth pressure. 2–4× speedup on CPU, 1.5–2× on GPU.
- Why: This matters because it tells you how to reason about quantization.
- Pitfall: Don't answer "Quantization" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: int8 inference reduces memory bandwidth pressure. 2–4× speedup on CPU, 1.5–2× on GPU.

### Distillation
- Direct Answer: train a smaller student model (4-layer BERT instead of 12-layer). 3–5× speedup with 2–3% quality loss.
- Why: This matters because it tells you how to reason about distillation.
- Pitfall: Don't answer "Distillation" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: train a smaller student model (4-layer BERT instead of 12-layer). 3–5× speedup with 2–3% quality loss.

### Pruning
- Direct Answer: remove attention heads or MLP neurons with low gradient norms. Structured pruning gives real speedups; unstructured pruning does not.
- Why: This matters because it tells you how to reason about pruning.
- Pitfall: Don't answer "Pruning" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: remove attention heads or MLP neurons with low gradient norms. Structured pruning gives real speedups; unstructured pruning does not.

### ONNX + TensorRT compilation
- Direct Answer: graph optimization and kernel fusion. 1.5–3× improvement without model changes.
- Why: This matters because it tells you how to reason about onnx + tensorrt compilation.
- Pitfall: Don't answer "ONNX + TensorRT compilation" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: graph optimization and kernel fusion. 1.5–3× improvement without model changes.

### Precompute user and item embeddings in batch (hourly or daily). Online lookup is O(1) instead of O(feature computation).
- Direct Answer: Precompute user and item embeddings in batch (hourly or daily). Online lookup is O(1) instead of O(feature computation).
- Why: This matters because it tells you how to reason about precompute user and item embeddings in batch (hourly or daily). online lookup is o(1) instead of o(feature computation)..
- Pitfall: Don't answer "Precompute user and item embeddings in batch (hourly or daily). Online lookup is O(1) instead of O(feature computation)." by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Precompute user and item embeddings in batch (hourly or daily). Online lookup is O(1) instead of O(feature computation).

### Use in-process caching
- Direct Answer: cache recent lookups to avoid repeated Redis calls for the same user within a session.
- Why: This matters because it tells you how to reason about use in-process caching.
- Pitfall: Don't answer "Use in-process caching" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: cache recent lookups to avoid repeated Redis calls for the same user within a session.

### Two-stage architecture
- Direct Answer: fast ANN retrieval (10–20ms) to get 500 candidates, then a lighter ranker on candidates. The expensive model does not score all items.
- Why: This matters because it tells you how to reason about two-stage architecture.
- Pitfall: Don't answer "Two-stage architecture" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: fast ANN retrieval (10–20ms) to get 500 candidates, then a lighter ranker on candidates. The expensive model does not score all items.

### Asynchronous precomputation
- Direct Answer: compute recommendations for users who are likely to visit soon (e.g., users active in the last hour) in background, serve from cache.
- Why: This matters because it tells you how to reason about asynchronous precomputation.
- Pitfall: Don't answer "Asynchronous precomputation" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: compute recommendations for users who are likely to visit soon (e.g., users active in the last hour) in background, serve from cache.

### Onboarding quiz
- Direct Answer: ask the user to rate 5–10 items or select preferred categories. Build a content-based user profile immediately.
- Why: This matters because it tells you how to reason about onboarding quiz.
- Pitfall: Don't answer "Onboarding quiz" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: ask the user to rate 5–10 items or select preferred categories. Build a content-based user profile immediately.

### Demographic priors
- Direct Answer: use coarse user attributes (location, device type) to assign a population-level prior. "New users from this region typically engage with X."
- Why: This matters because it tells you how to reason about demographic priors.
- Pitfall: Don't answer "Demographic priors" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: use coarse user attributes (location, device type) to assign a population-level prior. "New users from this region typically engage with X."

### Popularity
- Direct Answer: show globally popular items while personalization data accumulates. Not ideal, but better than random.
- Why: This matters because it tells you how to reason about popularity.
- Pitfall: Don't answer "Popularity" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: show globally popular items while personalization data accumulates. Not ideal, but better than random.

### Content-based similarity
- Direct Answer: embed new items using their metadata (genre, tags, description, cast). Find existing items with similar embeddings. Recommend to users who interacted with those similar items.
- Why: This matters because it tells you how to reason about content-based similarity.
- Pitfall: Don't answer "Content-based similarity" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: embed new items using their metadata (genre, tags, description, cast). Find existing items with similar embeddings. Recommend to users who interacted with those similar items.

### Cold start item injection
- Direct Answer: mix a fraction of new items into all users' recommendations. Collect feedback to bootstrap the CF signal.
- Why: This matters because it tells you how to reason about cold start item injection.
- Pitfall: Don't answer "Cold start item injection" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: mix a fraction of new items into all users' recommendations. Collect feedback to bootstrap the CF signal.

### Context-aware popularity
- Direct Answer: a new movie in genre X should be shown to users with strong genre X preference, not to the general population.
- Why: This matters because it tells you how to reason about context-aware popularity.
- Pitfall: Don't answer "Context-aware popularity" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: a new movie in genre X should be shown to users with strong genre X preference, not to the general population.

### Run a content-based system initially
- Direct Answer: Run a content-based system initially
- Why: This matters because it tells you how to reason about run a content-based system initially.
- Pitfall: Don't answer "Run a content-based system initially" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Run a content-based system initially

### Track all user interactions with full logging
- Direct Answer: Track all user interactions with full logging
- Why: This matters because it tells you how to reason about track all user interactions with full logging.
- Pitfall: Don't answer "Track all user interactions with full logging" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Track all user interactions with full logging

### Switch to collaborative filtering after enough signal accumulates
- Direct Answer: Switch to collaborative filtering after enough signal accumulates
- Why: This matters because it tells you how to reason about switch to collaborative filtering after enough signal accumulates.
- Pitfall: Don't answer "Switch to collaborative filtering after enough signal accumulates" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Switch to collaborative filtering after enough signal accumulates

### Define the transition threshold
- Direct Answer: "when the CF model's offline AUC on held-out interactions exceeds content-based by X%, switch"
- Why: This matters because it tells you how to reason about define the transition threshold.
- Pitfall: Don't answer "Define the transition threshold" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: "when the CF model's offline AUC on held-out interactions exceeds content-based by X%, switch"

### Data drift
- Direct Answer: P(X) changes. The input distribution shifts. The model may still perform well if P(Y|X) is stable.
- Why: This matters because it tells you how to reason about data drift.
- Pitfall: Don't answer "Data drift" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: P(X) changes. The input distribution shifts. The model may still perform well if P(Y|X) is stable.

### Concept drift
- Direct Answer: P(Y|X) changes. The same input now has a different label. The model's learned function is no longer correct regardless of how the inputs are distributed.
- Why: This matters because it tells you how to reason about concept drift.
- Pitfall: Don't answer "Concept drift" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: P(Y|X) changes. The same input now has a different label. The model's learned function is no longer correct regardless of how the inputs are distributed.

### Data drift can be detected by monitoring input distributions (PSI, KS test). No labels required.
- Direct Answer: Data drift can be detected by monitoring input distributions (PSI, KS test). No labels required.
- Why: This matters because it tells you how to reason about data drift can be detected by monitoring input distributions (psi, ks test). no labels required..
- Pitfall: Don't answer "Data drift can be detected by monitoring input distributions (PSI, KS test). No labels required." by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Data drift can be detected by monitoring input distributions (PSI, KS test). No labels required.

### Concept drift can only be detected by monitoring prediction accuracy. This requires ground truth labels, which often arrive with a delay.
- Direct Answer: Concept drift can only be detected by monitoring prediction accuracy. This requires ground truth labels, which often arrive with a delay.
- Why: This matters because it tells you how to reason about concept drift can only be detected by monitoring prediction accuracy. this requires ground truth labels, which often arrive with a delay..
- Pitfall: Don't answer "Concept drift can only be detected by monitoring prediction accuracy. This requires ground truth labels, which often arrive with a delay." by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Concept drift can only be detected by monitoring prediction accuracy. This requires ground truth labels, which often arrive with a delay.

### Monitor prediction distributions
- Direct Answer: if the model's score distribution shifts substantially while input distributions remain stable, the model is responding differently to the same inputs.
- Why: This matters because it tells you how to reason about monitor prediction distributions.
- Pitfall: Don't answer "Monitor prediction distributions" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: if the model's score distribution shifts substantially while input distributions remain stable, the model is responding differently to the same inputs.

### Proxy signals
- Direct Answer: in fraud, use chargeback rates as a lagged label. In recommendations, use downstream engagement as a label proxy.
- Why: This matters because it tells you how to reason about proxy signals.
- Pitfall: Don't answer "Proxy signals" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: in fraud, use chargeback rates as a lagged label. In recommendations, use downstream engagement as a label proxy.

### Drift detectors
- Direct Answer: ADWIN (Adaptive Windowing) or DDM (Drift Detection Method) on incoming predictions or calibrated errors.
- Why: This matters because it tells you how to reason about drift detectors.
- Pitfall: Don't answer "Drift detectors" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: ADWIN (Adaptive Windowing) or DDM (Drift Detection Method) on incoming predictions or calibrated errors.

### Interpreting coefficients (regulatory requirements, fairness analysis)
- Direct Answer: Interpreting coefficients (regulatory requirements, fairness analysis)
- Why: This matters because it tells you how to reason about interpreting coefficients (regulatory requirements, fairness analysis).
- Pitfall: Don't answer "Interpreting coefficients (regulatory requirements, fairness analysis)" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Interpreting coefficients (regulatory requirements, fairness analysis)

### Stable coefficient estimates across different data samples
- Direct Answer: Stable coefficient estimates across different data samples
- Why: This matters because it tells you how to reason about stable coefficient estimates across different data samples.
- Pitfall: Don't answer "Stable coefficient estimates across different data samples" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Stable coefficient estimates across different data samples

### Feature selection based on coefficient magnitude
- Direct Answer: Feature selection based on coefficient magnitude
- Why: This matters because it tells you how to reason about feature selection based on coefficient magnitude.
- Pitfall: Don't answer "Feature selection based on coefficient magnitude" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Feature selection based on coefficient magnitude
