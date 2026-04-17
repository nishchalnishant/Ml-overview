# Practical ML Scenarios

This file is basically the interview version of:

Something is broken.
Now what?

These scenarios matter because many ML interviews stop caring about pure definitions and start checking whether you can:

- diagnose
- prioritize
- stabilize
- ship

Which is where real engineering begins.

---

# 1. Model Great in Training, Bad in Production

If a model has 95% training accuracy and collapses in production, do not romanticize it.

The likely suspects are:

- leakage
- train-serving skew
- drift
- unrealistic validation
- threshold mismatch

**Strong answer**

I would first compare the offline setup and live setup end to end: feature definitions, preprocessing, data freshness, population shift, and label assumptions. I would not retrain blindly before isolating the failure mode.

That is the answer of someone who has seen real systems misbehave.

---

# 2. Extreme Class Imbalance: Fraud Is 0.01%

The first rule:

Do not say accuracy.

That metric becomes almost comic here.

Better tools:

- precision-recall AUC
- recall at target precision
- cost-sensitive thresholds
- class weights
- focal loss
- review-queue aware metrics

**Short answer**

In extreme imbalance settings, I would optimize for the business cost of false negatives and false positives rather than overall accuracy.

That is exactly the right instinct.

---

# 3. RAG System Is Hallucinating

If your RAG system hallucinates, the fix is usually not:

> "just trust the model harder"

You want to inspect:

- retrieval quality
- chunking strategy
- reranking
- prompt constraints
- citation enforcement
- answer-verification logic

**Strong answer**

I would tighten the prompt, improve retrieval quality, require grounded citations, and add evaluation that checks whether the answer is actually supported by retrieved context.

That sounds practical and grounded.

---

# 4. Real-Time Recommendation System Has High Latency

Classic systems question.

Possible levers:

- smaller model
- quantization
- distillation
- better caching
- candidate retrieval stage
- lighter online ranking

**Azure/DevOps lens**

Do not only ask:

> "Can we add more compute?"

Also ask:

> "Can we avoid doing this work on the critical path?"

That is the stronger systems answer every time.

---

# 5. Cold Start in Recommendation

Cold start means the system has little or no interaction history.

That can happen for:

- new users
- new items

Common fixes:

- content-based features
- popularity priors
- onboarding questions
- metadata-rich embeddings

**Fashion analogy**

If a new user arrives and you know nothing about her clicks yet, you can still start with:

- preferred style
- colors
- occasion
- brands

That is the content-based bridge before collaborative behavior kicks in.

---

# 6. Concept Drift in Production

Concept drift means the rules changed.

Examples:

- fraudsters adapt
- market behavior changes
- users respond differently after product changes

What to do:

- monitor
- detect
- recalibrate
- retrain
- compare against fallback model

The key point:

Not every drop in accuracy is just "train on newer data."
You need to know what changed first.

---

# 7. Feedback Loops

Feedback loops happen when the model's own decisions influence future data.

Examples:

- recommender keeps showing popular items, so they become even more popular
- moderation model suppresses borderline content, so future labels get biased

Why it matters:

Your training data stops being a neutral reflection of reality.

It becomes partly shaped by the model itself.

**Strong line**

I would think carefully about exploration, counterfactual logging, and how policy decisions shape the future label distribution.

That is a very strong answer.

---

# 8. Dead ReLU

Dead ReLU means a neuron outputs zero for almost everything and effectively stops learning.

Typical causes:

- high learning rate
- bad initialization
- unlucky weight updates

Common fixes:

- lower learning rate
- LeakyReLU
- better initialization

Small issue, very common interview mention.

---

# 9. Vanishing Gradient

This usually shows up in deep or recurrent models.

Symptoms:

- early layers learn slowly
- long-range dependencies are missed

Fixes:

- ReLU-style activations
- residual connections
- normalization
- LSTM/GRU
- Transformers

If you connect that back to architecture choice, you sound stronger.

---

# 10. Multicollinearity

When features are highly correlated, coefficients in linear models can become unstable and hard to interpret.

Fixes:

- drop redundant features
- use ridge regularization
- use dimensionality reduction

Important nuance:

Multicollinearity hurts interpretation more than raw predictive power in many cases.

That is a good line to remember.

---

# 11. Scenario Rapid-Fire

## Small object detection is poor

Check:

- input resolution
- anchor design
- feature pyramid setup
- label quality

## Fraud model catches too little

Check:

- threshold
- recall tradeoff
- positive-class weighting
- label delay

## Recommendation system feels repetitive

Check:

- diversity constraints
- exploration
- popularity bias

## Time-series model looks amazing offline

Check:

- leakage
- split strategy
- forecast horizon realism

---

# 12. A Very Good Scenario Answer Formula

When asked a practical scenario, answer in this order:

1. name likely failure modes
2. say what you would inspect first
3. explain how you would validate the diagnosis
4. explain what fix you would try
5. mention monitoring or rollback

That structure is calm, clear, and senior.

---

# Quick Thought Experiment

A production model suddenly degrades.

What sounds stronger?

1. "Let's retrain immediately."
2. "Let's first identify whether the failure is caused by drift, skew, threshold mismatch, or serving inconsistency."

Correct answer:

The second one.

Because diagnosis before action is how you avoid making the wrong system fail faster.
