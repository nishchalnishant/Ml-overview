---
module: Interview Prep
topic: Llm
subtopic: Machine Learning Interviews
status: unread
tags: [interviewprep, ml, llm-machine-learning-interview]
---
# Machine Learning Interview — Complete Preparation Guide

---

## 1. What the Interview Loop is Actually Measuring

Every round is testing a specific *capability*, not a knowledge domain. Recognizing the capability being tested changes how you construct your answer.

| Round | Underlying capability | How weak answers are spotted |
| :--- | :--- | :--- |
| Recruiter screen | Can you communicate your work to a non-expert? Do you have a coherent direction? | Vague descriptions signal unclear thinking, not just poor communication |
| DSA / coding | Do you decompose problems systematically, or do you thrash? | Interviewers watch how you handle new constraints mid-problem, not just whether you reach a solution |
| ML coding | Can you translate math to working code without pausing to look things up? | Numeric instability bugs, wrong gradient flow — the bugs that only appear if you don't understand what the code *means* |
| ML theory | Do you understand *why* things work, or just *what* they are? | Follow-up questions probe one causal layer deeper. "What happens if you remove the $\sqrt{d_k}$ scaling?" separates the two populations |
| ML system design | Do you think like someone who has shipped systems that break in production? | Missing cold start, monitoring, rollback, or training-serving skew is a strong junior signal |
| Behavioral / project | Do you own outcomes or just participate in them? | Passive framing — "the team did X" vs "I decided X and here's why" — is an ownership signal |

**Calibrate prep time by level:**
- L3 / junior: 60% ML coding + theory, 30% system design foundations, 10% behavioral
- L4 / mid: 40% system design, 35% theory depth, 25% behavioral
- L5+ / senior: 50% system design + failure modes, 30% behavioral with strong ownership framing, 20% theory depth

---

## 2. The Three-Layer Framework for Any Theory Question

Every ML theory question has three answerable layers. Stopping at layer 1 is a junior answer regardless of whether the definition is correct.

```
Layer 1 — Definition + formula
    What is it precisely? Include the math when relevant.

Layer 2 — Intuition: why this mechanism, not something simpler?
    What problem does it solve? What breaks if you remove it?
    Make the mechanism obvious in retrospect.

Layer 3 — Tradeoffs and failure modes
    When does it fail? What does it interact badly with?
    What's the alternative, and when would you choose the alternative?
```

**Applied to four common questions:**

**"Why BatchNorm?"**
- L1: Normalizes activations over the batch dimension per feature: $\hat{x} = (x - \mu_B)/\sigma_B$, then rescales with learnable $\gamma, \beta$.
- L2: Without it, each layer sees inputs whose distribution shifts as the preceding layers change during training — called internal covariate shift. The layer must simultaneously learn its task *and* adapt to a moving input distribution. BatchNorm fixes the input distribution, allowing a higher learning rate and faster convergence.
- L3: Breaks at small batch sizes (noisy $\mu_B, \sigma_B$ estimates corrupt the normalization). Breaks for variable-length sequences (different valid positions per sample — batch statistics are meaningless). Interacts badly with dropout (dropout changes the effective batch statistics at inference if ordering matters). For these cases: LayerNorm normalizes over the feature dimension per sample — sequence-length agnostic, works at batch size 1.

**"Why ReLU over sigmoid?"**
- L1: $\text{ReLU}(x) = \max(0, x)$; gradient is 1 for positive inputs, 0 for negative.
- L2: Sigmoid saturates at both ends — its derivative $\sigma(x)(1-\sigma(x))$ approaches 0 for large positive or negative inputs. Backpropagating through 20 layers of sigmoid multiplies gradients by $< 0.25^{20} \approx 10^{-13}$ — effectively zero. ReLU has gradient exactly 1 for positive pre-activations, so gradients don't shrink through ReLU layers.
- L3: Dying ReLU problem — if a neuron's pre-activation is always negative (e.g., from a bad initialization or a large weight update), the gradient is always 0 and the neuron never updates. Fix: LeakyReLU ($0.01x$ for $x < 0$) or careful He initialization. For Transformers, GELU is preferred — smooth approximation of ReLU that performs better on language tasks.

**"L1 vs L2 regularization?"**
- L1: $\lambda\sum|w_i|$; L2: $\lambda\sum w_i^2$.
- L2: The penalty on large weights is dominated by the gradient-of-squared-term, which shrinks weights proportionally. Some weight may get small but never exactly zero — the gradient at $w = 0$ is $2\lambda w = 0$, so no further push.
- L1: The penalty is $\lambda \cdot \text{sign}(w)$ — a constant push toward zero regardless of magnitude. For small nonzero weights, this push exceeds any gradient from the data, forcing them exactly to zero. This is why L1 produces sparse solutions. Probabilistic interpretation: L2 = Gaussian prior on weights (MAP), L1 = Laplace prior (fatter tails near zero — encourages sparsity).
- Tradeoff: L1 for feature selection and interpretability. L2 when all features are plausibly relevant. ElasticNet ($\alpha L1 + (1-\alpha) L2$) for both sparsity and stability.

**"Why cross-entropy over MSE for classification?"**
- L1: CE: $-\sum y_k \log \hat{p}_k$; MSE: $\sum(y - \hat{p})^2$.
- L2: Near the decision boundary, the MSE gradient with respect to the logit is small because the sigmoid's derivative is small there. You're in the middle of learning, and the gradient provides almost no update signal. Cross-entropy's gradient with respect to the logit is $\hat{p} - y$ — large and informative when the prediction is wrong, regardless of where you are in probability space.
- L3: MSE assumes Gaussian noise around the target, which is appropriate for regression. Classification outputs are Bernoulli (binary) or Categorical — cross-entropy is the correct log-likelihood for these distributions. Using MSE for classification is using the wrong probabilistic model.

---

## 3. How Depth Changes by Level

The same topic produces very different questions at different levels. Recognize which depth you're being evaluated at:

**Level 1 (L3 — correct definition):** "What is dropout?"
> "Randomly zeros activations with probability $p$ during training; surviving activations are scaled by $1/(1-p)$. At inference, all activations are used at full scale."

**Level 2 (L4 — reasoning about tradeoffs):** "Why use dropout vs L2 regularization?"
> "Both prevent overfitting, but through different mechanisms. L2 penalizes large weights — all neurons remain active but with shrunk weights; the model can still rely on any feature, just less strongly. Dropout randomly removes neurons during each training step — each neuron must learn to be useful in isolation, preventing neurons from co-adapting to each other's errors. Dropout approximates an ensemble of $2^n$ subnetworks with shared weights. The practical tradeoff: don't use both heavily together. L2 already shrinks weights; then dropout's rescaling creates compound uncertainty. With BatchNorm, use dropout *before* the BN layer — dropout after BN distorts the batch statistics that BN relies on."

**Level 3 (L5 — production judgment):** "We're seeing overfitting on a new domain after fine-tuning. Dropout is already at 0.3. What do you change and in what order?"
> "First, distinguish overfitting from distribution shift — they look identical in offline metrics. Check the validation loss trajectory: does it start high from epoch 1 (shift — the model never fits this distribution) or does it rise after initially decreasing (overfitting — the model starts fitting then memorizes)? If true overfitting: freeze the bottom 80% of layers and fine-tune only the top layers. The bottom layers contain general representations that shouldn't change; only the task-specific top layers need adaptation. If that's insufficient, increase dropout to 0.4-0.5 on the fine-tuning layers. If distribution shift: no amount of regularization helps — you need more representative data from the target domain. Mixing a small fraction (5-10%) of the original pretraining distribution prevents catastrophic forgetting of general representations."

---

## 4. ML Coding — What Must Be Automatic

These must be writeable with zero hesitation. Each one has a numerical stability failure mode. Full implementations with reasoning: [ml-coding-patterns.md](05-ml-coding-patterns.md).

**The six patterns you must write from scratch:**

1. **Sigmoid + BCE loss** — numerically stable form: clip z before `exp` or branch on sign. BCE needs `eps` to prevent `log(0)`. See [ml-coding-patterns.md §3](05-ml-coding-patterns.md).

2. **Numerically stable softmax** — shift by `max(x)` before exponentiation. Invariance proof: `exp(x_i - c) / Σ exp(x_j - c) = exp(x_i) / Σ exp(x_j)`. See [ml-coding-patterns.md §1](05-ml-coding-patterns.md).

3. **Precision / Recall / F1** — TP, FP, FN from scratch. Denominator guards for zero division. F1 = harmonic mean punishes P/R imbalance harder than arithmetic mean. See [ml-coding-patterns.md §5](05-ml-coding-patterns.md).

4. **Scaled dot-product attention** — `scores = Q @ K.T / sqrt(d_k)`, apply causal mask before softmax (`-1e9` for masked positions), then `weights @ V`. See [ml-coding-patterns.md §8](05-ml-coding-patterns.md).

5. **K-Means E+M step** — E: `argmin` over centroid distances. M: mean of assigned points, keep old centroid if empty cluster. See [ml-coding-patterns.md §4](05-ml-coding-patterns.md).

6. **PyTorch training loop bugs** — four bugs to know: (1) missing `zero_grad()` accumulates gradients; (2) gradient clipping must precede `optimizer.step()`; (3) inference requires both `model.eval()` (disables dropout/BN train mode) and `torch.no_grad()` (stops autograd graph); (4) checkpoint must include optimizer state — losing Adam's m/v buffers loses momentum. See [ml-coding-patterns.md §7](05-ml-coding-patterns.md).

---

## 5. ML System Design — Universal Structure

Never start with a model. The model is the *last* thing you choose. Full 10-step framework with worked examples: [ml-system-design.md](04-ml-system-design.md).

**Step summary:** Clarify goal → Metrics (offline + online) → Constraints (latency/QPS/budget) → Data + labels (freshness, delay, skew) → Baseline → Architecture (justify vs constraints) → Feature pipeline (real-time vs batch, fit on train only) → Serving (batch vs real-time, caching) → Evaluation (A/B design, backtesting) → Monitoring + rollback (PSI >0.25 = retrain, rollback defined before launch).

**Signals that distinguish senior responses:**

| Signal | What it reveals about experience |
| :--- | :--- |
| Cold start handling | Every production system has new users and new items. Missing this is a strong junior signal. |
| Feedback loops | Recommending popular items makes them more popular. Self-reinforcing bias is a real design problem, not a theoretical one. |
| Rollback plan | Rollback should be defined before launch. Defining it after something breaks means you shipped without a safety net. |
| Cost-aware threshold | Default 0.5 threshold is almost never optimal. The threshold should come from the cost matrix $C_{FP}, C_{FN}$. |
| Training-serving skew | The most common source of offline-online metric divergence. Missing this means you'd waste days debugging the wrong thing. |
| Ground truth delay | For fraud, chargebacks arrive 30-60 days after the transaction. Your "current" labels are a month old. |

---

## 6. Behavioral / Project Deep Dive

Every project story requires seven elements. Missing any one costs you.

| Element | Weak version | Strong version |
| :--- | :--- | :--- |
| Problem | "We needed better recommendations" | "CTR on homepage dropped 8% after a catalog expansion — our model had no cold start handling for new items, so new items never appeared in recommendations" |
| Constraints | (omitted) | "< 50ms P99 latency, retraining budget of once per week, team of 2 MLE, model decisions must be explainable to legal" |
| Approach | "We used a two-tower model" | "Started with matrix factorization as a baseline (2 days). Per-segment analysis showed cold start drove most of the gap. Added a content-based two-tower for new items only, routing traffic by item age" |
| Tradeoffs | (omitted) | "Two-tower adds 40ms latency vs MF. Accepted this because it recovered the 8% CTR gap on new items. Recall@100 on existing items dropped 3% — accepted because the business impact was smaller" |
| Metrics | "AUC improved" | "Offline NDCG@10 +12%. A/B: CTR +6.3%, revenue/user +2.1%, p < 0.01 at n = 500K per arm" |
| Production outcome | "We shipped it" | "5% canary → 50% → 100% over two weeks. One rollback event: feature pipeline failure detected at 5% canary — zero user impact because we hadn't scaled up" |
| Lesson | (omitted) | "We didn't run cold start analysis before building the baseline. Cold start evaluation is now a first-day checklist item for every new model." |

**Three story types you must have prepared, not just one:**
1. **Technical problem-solving:** you diagnosed a hard problem systematically — unexpected degradation, a confounding variable, a bug that looked like a model problem but was a data problem
2. **Failure story:** you shipped something wrong, or made a judgment call that turned out to be incorrect — what happened, what you did, what changed
3. **Tradeoff / disagreement:** you chose the slower or more conservative option over stakeholder pressure, with a quantified reason

---

## 7. Key Tradeoffs — the "Why X and Not Y" Pattern

The right answer to any tradeoff question is: "it depends on these factors, and here is how I would decide."

| Tradeoff | Decision logic |
| :--- | :--- |
| XGBoost vs neural net | XGBoost for tabular data < 1M rows — right inductive bias, fast iteration, interpretable. Neural net when jointly embedding high-cardinality IDs or raw text, sharing representations across tasks, or dataset > 10M rows. The data size and embedding requirement are the decision variables. |
| Precision vs recall | Depends on the cost matrix: $C_{FN}$ vs $C_{FP}$. Fraud: missed fraud ($C_{FN}$) >> false block ($C_{FP}$) → recall-oriented. Spam: false block ($C_{FP}$) >> missed spam ($C_{FN}$) → precision-oriented. The threshold should minimize $\text{FP} \times C_{FP} + \text{FN} \times C_{FN}$, not maximize F1. |
| BatchNorm vs LayerNorm | BatchNorm normalizes over the batch dimension — requires a large, consistent batch; fails for variable-length sequences; breaks at batch size 1. LayerNorm normalizes over features per sample — sequence-length agnostic, standard for Transformers, works at any batch size. RMSNorm (LLaMA): skips mean subtraction (empirically unimportant), faster. |
| L1 vs L2 | L1: sparse — some weights exactly 0. Use for feature selection and interpretability. L2: all weights small, none exactly 0. Use when all features are plausibly relevant. ElasticNet: use when you want both sparsity and stability. |
| Online vs batch serving | Online when the decision depends on current context (fraud at transaction time, autocomplete). Batch when freshness is not required within the latency budget (weekly churn scores). Hybrid is most common in production: pre-compute embeddings offline, assemble and score at query time. |
| SMOTE vs class_weight | Try `class_weight='balanced'` first — free, no data generation, works for most models. SMOTE for severe imbalance in tabular data when the minority class is genuinely underrepresented in coverage, not just count. Neither fix handles distribution shift — that requires domain adaptation or more representative data. |

---

## 8. Numbers to Know Without Thinking

| Fact | Value |
| :--- | :--- |
| GPT-3 parameters | 175B |
| Chinchilla optimal tokens per parameter | ~20 |
| LLaMA 3 training tokens | 15T |
| Standard attention complexity | $O(n^2 d)$ |
| LoRA trainable parameters (typical) | ~0.06% of base model |
| BF16 memory per parameter | 2 bytes |
| FP32 memory per parameter | 4 bytes |
| 70B model in BF16 VRAM | ~140GB |
| PSI retrain threshold | > 0.25 |
| AdamW default $\beta_1, \beta_2$ | 0.9, 0.999 (LLM pretraining: 0.9, 0.95) |
| Warmup fraction of training | 1–5% of total steps |
| KV cache per 1K tokens (LLaMA 3 70B) | ~160MB |

---

## 9. Mistakes That Cost You the Role

| Mistake | What it signals |
| :--- | :--- |
| Starting system design with model choice | No requirements discipline — you'd design the right model for the wrong problem |
| Mentioning only accuracy as a metric | No understanding of class imbalance or business impact |
| Skipping cold start in a recommendation design | Junior signal — every production recommendation system encounters this |
| "Retrain the model" as first response to a production accuracy drop | No debugging instinct — you'd waste days on the wrong fix |
| Not mentioning monitoring or rollback | You think shipping = done; no production experience |
| Confident misinterpretation of p-values | Statistics red flag — "p < 0.05 means the null is probably false" is wrong |
| Rote behavioral answers with passive framing | "The team did" instead of "I decided because" — low ownership signal |

---

## 10. Pre-Interview Checklist

**Must write from scratch, zero hesitation:**
- [ ] Sigmoid, softmax, numerically stable BCE
- [ ] Precision, recall, F1 with correct denominators stated verbally
- [ ] Bias-variance decomposition formula + one-sentence interpretation of each term
- [ ] Bayes theorem applied to the rare-disease test example (the answer is ~9%, not 99%)
- [ ] Correct p-value interpretation + two wrong interpretations stated explicitly
- [ ] Two-stage retrieval + ranking sketch
- [ ] The 10-step system design framework

**Must explain tradeoffs fluently:**
- [ ] L1 vs L2 vs Dropout vs Early stopping (mechanism, not just effect)
- [ ] BatchNorm vs LayerNorm — with the BatchNorm failure conditions
- [ ] Batch vs real-time serving — with the hybrid pattern
- [ ] Precision vs recall as a cost matrix problem
- [ ] XGBoost vs neural net — with specific decision criteria

**Must sketch from memory:**
- [ ] Two-tower recommendation system (retrieval + ranking + cold start path)
- [ ] Fraud detection pipeline (Redis velocity → rule engine → ML model → cost-aware threshold)
- [ ] LLM serving stack (load balancer → vLLM → KV cache → streaming)
- [ ] Temporal train/val/test split — why random splits fail for time-series

---

## 11. Company-Specific ML Interview Prep (FAANG)

Each company has structural differences in how they evaluate ML candidates. Knowing the format in advance removes one source of uncertainty.

### Meta (Facebook)

**Format:** 2 coding rounds + 1 ML design + 1 behavioral (for L4+: 2 ML design rounds).

**Coding:** LeetCode medium-hard on graphs, dynamic programming, and string manipulation. ML coding (from scratch): gradient descent, softmax, precision/recall. Meta tests implementation speed — you have ~35 minutes per problem.

**ML Design:** The signature question is product-oriented: "Design the feed ranking system," "Design Ads CTR prediction," "Design a content moderation classifier." Meta expects you to start from business metrics (ranking by engagement, click-through rate, policy violations per mille), propose two-stage retrieval + ranking, and discuss feedback loop risks explicitly. Cold start, filter bubbles, and A/B testing design are all expected.

**Known focus areas:**
- Two-tower models for retrieval; GBDT (LightGBM) or wide-and-deep for ranking
- Integrity ML: anomaly detection, graph-based abuse detection, content classifiers
- Ads ML: calibration (prediction of P(click) must be calibrated, not just rank-ordered)
- Real-time feature freshness: how stale can embeddings be? How do you handle recency?

**Behavioral:** Meta's values are "Move Fast," "Focus on Long-Term Impact," "Be Direct," and "Build Social Value." Stories should show impact at scale. Avoid passive framing — "the team decided" is a red flag.

---

### Google / DeepMind

**Format:** 4–5 rounds: 1–2 ML theory, 1–2 coding (algorithms + ML coding), 1 system design (ML or general), 1 behavioral/Googleyness.

**Theory depth:** Google ML interviews go deeper on math than Meta or Amazon. Expect derivation-level questions: "Derive the gradient of cross-entropy with respect to logits," "Explain why Adam's bias correction is necessary in early steps," "What does the Fisher information matrix have to do with natural gradient descent?"

**ML Design:** Google expects architectural justification against constraints more than any other company. "Why two-stage instead of single-stage? What QPS does the ranker need to handle? How does your monitoring trigger a rollback?" Google has a strong preference for candidates who acknowledge uncertainty and propose measurable validation checkpoints.

**Known focus areas:**
- Search ranking: LTR (Learning to Rank), NDCG, position bias correction, counterfactual evaluation
- YouTube RecSys: watch-time optimization, serendipity vs relevance tradeoff, diversity
- LLM safety and alignment (for research-adjacent roles)
- Infrastructure: TensorFlow/JAX proficiency, TPU serving patterns

**Coding:** Google uses custom OA and live interviews. Medium–hard algorithms. ML coding: attention from scratch, backprop through a custom layer, implementing a sampling strategy.

---

### Amazon / AWS

**Format:** "Loop" interview: 5–7 rounds on the same day, each led by a different interviewer. Each interviewer owns a Leadership Principle (LP). One round is typically ML design, one is coding, the rest are behavioral anchored to LPs.

**Leadership Principles:** Amazon cares about LPs as much as technical skill. For ML roles, the most tested: Customer Obsession (how did ML decisions affect user experience?), Dive Deep (show you understand your system's internals), Invent and Simplify (how did you reduce ML pipeline complexity?), Bias for Action (shipped something under uncertainty with a clear rollback plan).

**ML Design:** More operations-oriented than Meta/Google. "Design a product recommendation system for Amazon," "Design a fraud detection system for Amazon Payments." Cost-benefit analysis of model complexity is expected: "Would a simpler model save $X in compute and lose $Y in revenue? What's the tradeoff?"

**Known focus areas:**
- Demand forecasting (DeepAR, LSTMs for time series)
- Search ranking + personalization at Amazon scale
- Fraud and abuse detection (cost-aware thresholds, online learning)
- MLOps and SageMaker patterns for applied roles

---

### Apple

**Format:** Multiple rounds, typically on-site over one or two days. Mix of ML theory, coding, and system design. Apple is more secretive about format than other companies — expect variability.

**Character of questions:** Apple ML interviews tend toward applied rather than pure research. Expect questions like "How would you improve Siri's NLU?" or "How would you reduce latency in our on-device model?" Privacy is a genuine first-class concern — on-device ML, federated learning, and differential privacy appear more at Apple than elsewhere.

**Known focus areas:**
- On-device inference optimization: quantization, distillation, CoreML
- Federated learning and privacy-preserving ML (differential privacy, secure aggregation)
- Siri and NLP: intent classification, dialogue management, ASR pipelines
- Computer vision: Face ID, photo organization, object detection on mobile

**Coding:** Expect both algorithms and ML coding. Proficiency with Swift/CoreML is a plus for applied roles.

---

### Microsoft / Azure ML

**Format:** 4–5 rounds: coding, ML design, behavioral. For research roles, may include a presentation or paper discussion.

**Character:** Microsoft ML roles span a wide range — Azure ML platform, Bing search, Copilot/LLM integration, and gaming (Xbox). Interview flavor depends heavily on the team. Platform and infrastructure roles weight MLOps and system reliability heavily. Product-facing roles weight user impact and A/B testing.

**Known focus areas:**
- Azure ML + MLflow: model registry, pipeline design, experiment tracking
- Responsible AI: fairness metrics, bias detection, model cards
- OpenAI partnership roles: LLM fine-tuning, RLHF, RAG architecture
- Bing ranking: query understanding, LTR, relevance evaluation

**Behavioral:** Microsoft uses "Situation, Task, Action, Result" with emphasis on collaboration and "growth mindset" (not just what you did, but what you learned).

---

### General Cross-Company Patterns

| Dimension | Meta | Google | Amazon | Apple | Microsoft |
|-----------|------|--------|--------|-------|-----------|
| LP/Values emphasis | High | Medium | Very high | Low | Medium |
| Theory depth | Medium | High | Low | Medium | Medium |
| System design weight | High | High | High | Medium | High |
| Coding difficulty | Medium-Hard | Hard | Medium | Medium | Medium |
| Production/ops focus | High | Medium | High | Very high (on-device) | High |
| Privacy/ethics topics | Medium | Medium | Low | Very high | High |

**Tips that apply to all:**
- Never say "we used X model" without a justification grounded in constraints. Always: "we chose X over Y because of constraint Z."
- At L5+, the bar is whether you can define the problem more sharply than the interviewer stated it. Reframe is a senior signal.
- Interviewers take notes on what you say without prompting (cold start, rollback, feedback loops) vs what you say only when asked. The former is the senior bar.

---

## 12. Take-Home Case Study Tips

Take-home assignments are common at L4+ and for applied scientist roles. They are treated differently from live interviews — the bar is higher because you have time, but the evaluation is also more revealing.

### What Evaluators Actually Look For

Take-homes are not about perfection. They test:
1. **Problem framing** — did you identify the actual business problem, not just run models on the data?
2. **Evaluation rigor** — did you choose metrics appropriate to the problem, or did you optimize accuracy on a balanced holdout?
3. **Tradeoff awareness** — did you acknowledge what your approach can't do?
4. **Reproducibility** — can the evaluator reproduce your results in under 5 minutes?
5. **Communication** — can you explain technical choices to a non-ML reader in a 3-line summary?

### Structured Approach

**Step 1: Understand the evaluation criteria before writing code.** Read the instructions twice. Identify: what success metric do they care about? Is there a class imbalance trap? Is there a temporal aspect that would invalidate random splits?

**Step 2: Establish a strong baseline first.** A logistic regression or random forest baseline that works correctly is worth more than a neural net that barely beats it. Evaluators are checking whether you understand why the baseline fails before you improve on it.

**Step 3: Make your EDA communicate.** Limit EDA to 3–5 plots that directly inform modeling decisions. "Class distribution → chose PR-AUC over accuracy." "Feature X has 40% missing rate → imputation strategy." Plot only what changed your decisions.

**Step 4: Explicitly justify every modeling choice.** Don't just try things. State: "I chose LightGBM over logistic regression because the relationship between `user_age` and `churn` is non-monotonic (shown in EDA §2), and LightGBM handles this without manual feature engineering."

**Step 5: Discuss what you'd do next, with priorities.** "With more time, I'd: (1) add temporal cross-validation because train/test split may have leakage on daily aggregated features, (2) tune the threshold using cost matrix C_FP=1, C_FN=5 rather than default 0.5, (3) add a SHAP analysis to validate that model decisions are driven by business-meaningful features."

### Common Take-Home Mistakes

- **Data leakage from the future** — joining on a timestamp without respecting point-in-time correctness. Features computed from events that happen after the label timestamp inflate model performance and fail in production.
- **Optimizing AUC when the task is threshold-sensitive** — if the deliverable is a classifier, report precision and recall at your chosen threshold, not just AUC.
- **Over-engineering** — a neural network where logistic regression would work equally well signals poor judgment, not skill.
- **No uncertainty quantification** — state confidence intervals or standard deviations on your metrics. A model that achieves 0.82 AUC with std 0.08 across folds is not better than 0.80 with std 0.01.
- **Not checking for class imbalance** — always report class counts in your EDA. Missing this is a top-5 automatic disqualifier.
- **Notebooks that require manual cell ordering** — use `Run All` on a clean kernel as your final check. If it fails, it fails the evaluator too.
