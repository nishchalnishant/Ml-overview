---
module: Practice
topic: System Design Mocks
subtopic: ML Platform
status: unread
tags: [practice, system-design, mock-interview, ml-platform]
---
# ML Platform Mock Scripts

Four scripts. Each runs 45 minutes. The worked *designs* live in
[15-system-design/cases/](../../15-system-design/cases/) — these scripts supply the
**pressure**, which is what the round is actually scored on.

**Read only the prompt block first.** Open each stage at the stated minute mark.

---

## Mock 1 — Real-time feature store

> **Prompt (read at 0:00).** We have twenty ML teams. Each one is building its own
> feature pipelines, and we keep discovering that the features used in training differ
> from the ones computed at serving time. Design a feature store that fixes this.
> Assume 500 features, 50M entities, 100k prediction requests per second at peak.

<details>
<summary><b>Stage 1 — open at 0:15</b></summary>

Ask, in order, and do not accept a hand-wave on any of them:

1. "Walk me through exactly how a feature gets from raw event to the model at serving
   time. Where in that path does skew get introduced?"
2. "You said batch and streaming. Is the aggregation code the same code in both, or two
   implementations? If two, how do you know they agree?"
3. "What is your p99 for a 500-feature lookup, and where does the time go?"

**What you are listening for:** the candidate must land on a *single definition* of each
feature that both paths execute, and must name point-in-time correctness for training
data generation. If they only say "we store features in Redis," they have designed a
cache, not a feature store.
</details>

<details>
<summary><b>Stage 2 — open at 0:25</b></summary>

> "A data scientist tells you their model's offline AUC is 0.91 but production is 0.68.
> They swear the code is identical. Using your design, how do you find the cause?"

Then, whatever they answer:

> "It turns out one feature is `days_since_last_purchase`. Explain how that feature
> specifically could break, and how your design prevents it."

**Listening for:** the point-in-time join. Computing `days_since_last_purchase` for a
training row using *today's* purchase table rather than the value as of the label
timestamp is the canonical training-serving skew, and a good candidate reaches for it
immediately.

**Calibration:**
- *Weak:* "I'd add logging and compare." (Reactive; no mechanism.)
- *Solid:* "Point-in-time correct joins on the offline store, and I'd log served
  feature vectors so I can replay production inputs through the offline path."
- *Strong:* the above, plus "I'd make the offline path physically unable to see future
  rows — the training dataset builder takes the label timestamp as a required argument,
  so a leaky join is a type error rather than a code review catch."
</details>

<details>
<summary><b>Stage 3 — open at 0:35</b></summary>

> "Redis just went down in one region. What happens to the 100k rps of predictions?"

Follow with:
> "Your team says we should fall back to computing features from the source database.
> What's wrong with that?"

**Listening for:** graceful degradation over correctness-at-any-cost. Falling back to
the source DB shifts 100k rps of point queries onto an OLTP system that will fail, taking
the outage from one service to two. Acceptable answers: serve stale features with a
staleness flag the model can consume, serve a default vector and a fallback model, or
fail open to a heuristic. The strong answer notes that a model receiving stale features
without being *told* they are stale is worse than a model that knows.
</details>

<details>
<summary><b>Rubric — see <a href="03-rubrics.md">03-rubrics.md</a></b></summary>

Weight this one heavily on **correctness under time** (point-in-time joins) and
**failure reasoning**. Scale is not the hard part here and should not dominate.
</details>

---

## Mock 2 — Training pipeline for a 100-person org

> **Prompt (read at 0:00).** Design the training infrastructure for an organization
> running roughly 200 model training jobs per day, ranging from a five-minute
> gradient-boosting fit to a three-day multi-GPU fine-tune. Requirements: any result must
> be reproducible six months later, and a single team must not be able to starve the
> others of GPUs.

<details>
<summary><b>Stage 1 — open at 0:15</b></summary>

1. "Define reproducible. What exactly do you have to pin?"
2. "A job fails 60 hours into a 72-hour run. What happens?"
3. "How does a job get a GPU? What decides the order?"

**Listening for:** reproducibility is code SHA *plus* data snapshot/version *plus*
environment image *plus* hyperparameters *plus* random seeds *plus* library and CUDA
versions. Candidates who list only code and hyperparameters have not shipped this.
Checkpointing must come up unprompted for the 72-hour case.
</details>

<details>
<summary><b>Stage 2 — open at 0:25</b></summary>

> "The research team submits fifty short jobs. The platform team has one three-day job
> queued. Under your scheduler, what happens, and is that the right outcome?"

Then:
> "Finance says GPU spend doubled last quarter and nobody knows who's responsible.
> Fix that."

**Listening for:** the difference between fair-share and FIFO, gang scheduling for
multi-GPU jobs (a distributed job holding four of eight needed GPUs while waiting is a
deadlock, and the candidate should see it), preemption with checkpointing as the release
valve, and chargeback requiring per-job attribution tagging designed in from the start
rather than reconstructed from billing exports.
</details>

<details>
<summary><b>Stage 3 — open at 0:35</b></summary>

> "Someone needs to reproduce a model from eight months ago for a regulatory audit. The
> training data lived in a table that gets updated in place. Walk me through it."

**Listening for:** honesty. If the design mutates tables in place, the correct answer is
"with the design as stated, I cannot — and that is a defect I would fix with immutable
snapshots or a versioned table format like Delta/Iceberg." A candidate who invents a
recovery path that their own design does not support is a red flag; one who identifies
the gap in their own design is a strong signal.
</details>

---

## Mock 3 — Recommendation system

> **Prompt (read at 0:00).** Ten million users, two million items, and we want a
> personalized home feed. 5,000 requests per second, p99 under 150 ms end to end. Design
> it.

<details>
<summary><b>Stage 1 — open at 0:15</b></summary>

1. "You cannot score two million items in 150 ms. What is the actual architecture?"
2. "How does a brand-new item ever get shown to anyone?"
3. "What are you optimizing? Give me the label."

**Listening for:** retrieval-then-ranking, with a candidate-generation stage cutting
2M → ~1000 and a heavier ranker over those. If they propose scoring everything, that is
the round's first real problem and they should catch it themselves. Cold start must be
addressed for both new items and new users, and they are different problems. On the
label: clicks are the easy answer and the interesting follow-up is that optimizing
clicks produces clickbait — the strong candidate raises the proxy-metric mismatch
without prompting.
</details>

<details>
<summary><b>Stage 2 — open at 0:25</b></summary>

> "Your ranker takes 40 ms for 1,000 candidates. Product wants 5,000 candidates. Options?"

Then:
> "We shipped it. Engagement is up 8%, but the content team says the feed has become
> repetitive and creators with small audiences have stopped getting impressions. What
> now?"

**Listening for:** on latency — batching, a cheaper first-pass ranker, quantization,
truncating candidates by a fast pre-score, or accepting a p99 increase and negotiating.
On the second question the answer is *not* purely technical: this is a feedback loop
where the model's own outputs become its training data, popularity bias compounds, and
the fix combines diversity constraints in ranking, exploration budget for tail items,
and a metric beyond engagement. A candidate who only proposes "add a diversity penalty"
without naming the feedback loop is at solid, not strong.
</details>

<details>
<summary><b>Stage 3 — open at 0:35</b></summary>

> "How do you know the new model is better than the old one, before you ship it to
> everyone?"

Push on whatever they say:
> "Your A/B test shows +2% engagement with p = 0.04. Ship it?"

**Listening for:** offline metrics gate but do not decide; online A/B decides. On the
p-value question: how many variants were tested (multiple comparisons), what was the
pre-registered primary metric, what is the minimum detectable effect and was the test
powered for it, did guardrail metrics move, and how long did it run relative to weekly
seasonality. "p < 0.05, ship it" is the weak answer.
</details>

---

## Mock 4 — Fraud detection

> **Prompt (read at 0:00).** Payments platform, 50,000 transactions per second at peak.
> Roughly one in two thousand is fraudulent. You must approve or decline within 100 ms.
> Design the system.

<details>
<summary><b>Stage 1 — open at 0:15</b></summary>

1. "What is your metric, and why is accuracy not it?"
2. "Where does the label come from, and when does it arrive?"
3. "100 ms budget — allocate it."

**Listening for:** PR-AUC or recall at fixed precision; the cost asymmetry between a
missed fraud and a declined legitimate customer must be raised explicitly and ideally
quantified. On labels: chargebacks arrive weeks to months later, so the training label
is severely delayed and partially unobservable — declined transactions never generate a
label at all, which is a selection-bias problem the strong candidate names as such.
</details>

<details>
<summary><b>Stage 2 — open at 0:25</b></summary>

> "Fraud rings adapt within days of a rule going live. Your model retrains weekly.
> What's your answer to that?"

Then:
> "You retrain on production data. Your model declined a pattern last month, so you have
> no examples of it succeeding. What does that do to the next model?"

**Listening for:** layered defense — fast rules for known patterns, a model for
generalization, and human review for the uncertain band — plus fast retraining or online
learning for the adaptive part. The second question is the feedback loop again: the model
creates its own training distribution, and the mitigation is a small randomized
holdout that is allowed through (accepting real fraud loss as the price of unbiased
labels), plus explicit tracking of the declined population.
</details>

<details>
<summary><b>Stage 3 — open at 0:35</b></summary>

> "A regulator asks why a specific customer was declined. Answer them."

Then, if time:
> "Your model uses ZIP code and it improves AUC by 3%. Keep it?"

**Listening for:** per-decision explanation requires logging the feature vector and the
model version at decision time, not reconstructing it later — reason codes computed and
stored inline. On ZIP code: this is a proxy-for-protected-attribute question. The answer
is not automatically "drop it"; it is that you must measure disparate impact across
protected groups, and that a 3% AUC gain does not justify an unmeasured fairness cost.
See [14-responsible-ai/](../../14-responsible-ai/). A candidate who answers purely on
AUC has failed the question.
</details>
