# Model Evaluation

This file exists to save you from the most common ML interview disaster:

> "The model has 98% accuracy."

Lovely.
Now tell me whether that number means anything.

---

# 1. Evaluation = Quality Gates for Models

If training is your **build pipeline**, evaluation is your **release gate**.

A model is not "good" because it trained successfully.
A model is good if it survives the right checks:

- correct metric
- correct split
- correct threshold
- correct business context

If any one of those is wrong, the model can look brilliant and still be useless.

---

# 2. Accuracy, Precision, Recall, F1

## Accuracy

How often is the model correct overall?

Great when classes are balanced.
Dangerous when they are not.

If fraud is only 1% of transactions, a model predicting "not fraud" every time gets **99% accuracy**.
Which is technically impressive and practically embarrassing.

## Precision

Of all predicted positives, how many were truly positive?

Use when false alarms are expensive.

Examples:

- fraud alerts
- spam detection
- manual review queues

## Recall

Of all real positives, how many did we catch?

Use when missed cases are expensive.

Examples:

- cancer screening
- defect detection
- security breach alerts

## F1

The balanced score when precision and recall both matter.

**Cricket analogy**

- Precision = when you go for the big shot, how often does it land?
- Recall = of all hittable balls, how many did you actually convert?
- F1 = the batter who knows when to attack and when not to throw wicket away

---

# 3. Confusion Matrix

The confusion matrix is the scoreboard behind the metric.

It tells you:

- true positives
- false positives
- true negatives
- false negatives

Why it matters:

Because two models can have the same accuracy and very different failure patterns.

And in production, **failure pattern** matters more than bragging rights.

---

# 4. ROC-AUC vs PR-AUC

## ROC-AUC

Measures how well the model separates positives from negatives across thresholds.

Useful when classes are reasonably balanced.

## PR-AUC

More useful when the positive class is rare.

Because it focuses on:

- precision
- recall

which is usually what you actually care about in imbalanced problems.

**Easy interview line**

If positives are rare and important, PR-AUC is often more honest than ROC-AUC.

---

# 5. Log Loss vs Accuracy

Accuracy only cares if the final answer was right.

Log loss cares about **how confident** the model was.

That means:

- mildly wrong = not great
- confidently wrong = painful

This is why log loss matters for:

- ranking
- calibrated probability systems
- bidding
- risk scoring

**DevOps analogy**

Accuracy is like checking whether a deployment passed or failed.
Log loss is like also checking how close the deployment was to catastrophe even when it "passed."

---

# 6. Regression Metrics: MAE, MSE, RMSE

## MAE

Average absolute error.

Good when you want:

- interpretability
- robustness to outliers

## MSE

Squares the error.

Big mistakes get punished much more heavily.

## RMSE

Square root of MSE.

Same "big error penalty" behavior, but easier to interpret because it is back in the original unit.

**Fashion analogy**

If your outfit sizing prediction is off by 1 inch, okay.
If it is off by 7 inches, chaos.

RMSE and MSE are the metrics that say:

> "Big misses deserve extra consequences."

---

# 7. Choosing the Right Metric

This is where interview answers become senior-level.

Do not just say:

> "I would use F1."

Say:

> "I would choose the metric based on business cost and operational behavior."

Ask:

- Are false positives costly?
- Are false negatives costly?
- Do we need calibrated probabilities?
- Is this a ranking problem?
- Is class imbalance severe?

**Short rule**

- balanced classification = accuracy can be okay
- rare positive class = precision/recall/PR-AUC
- probability quality matters = log loss / Brier / calibration
- regression with big-error pain = RMSE
- regression with robust interpretation = MAE

---

# 8. Calibration

Calibration means:

If the model says "80% probability," then about 80 out of 100 such cases should really be positive.

This matters a lot in:

- fraud
- medical systems
- pricing
- recommendations
- ads

A model can rank well and still be badly calibrated.

That means:

- good ordering
- bad probabilities

Both are not the same thing.

---

# 9. Cross-Validation

Cross-validation is how you reduce dependence on one lucky split.

Instead of trusting one train/validation split, you rotate across folds.

That gives you a more stable estimate.

**Azure/DevOps parallel**

It is the difference between:

- validating one deployment path once

and

- validating across multiple environments and conditions

Would you trust only one deployment test?
Exactly.

---

# 10. Class Imbalance

Imbalanced data is where bad evaluation habits go to thrive.

Common fixes:

- use better metrics
- tune threshold
- class weighting
- resampling
- focal loss
- better features

The key idea:

Do not "solve" imbalance by only changing the training data.
Often the biggest fix is choosing the right metric and threshold first.

---

# 11. Offline vs Online Evaluation

Offline metrics are necessary.
They are not enough.

Why?

Because users are messy.

A model can improve offline and still hurt:

- CTR
- retention
- revenue
- user trust

So in production you often need:

- offline evaluation
- shadow testing
- canary rollout
- A/B testing

That will feel very natural if you come from DevOps.

This is just release management with smarter artifacts.

---

# 12. Recommendation Metrics

For recommenders, we often care about the top of the ranked list.

Useful metrics:

- Precision@K
- Recall@K
- MAP
- NDCG

These matter because users do not scroll forever.
The first few items do the heavy lifting.

**Mini Pop Quiz**

If the best item is ranked 50th, is the recommendation system good?

No.

Technically relevant is not the same as practically useful.

---

# 13. A/B Testing for ML

A/B testing compares model variants in live traffic.

Use it when you want to know:

- does this model improve the real business metric?
- does it create new failure modes?
- does it affect user behavior in surprising ways?

**Key terms to sound solid in an interview**

- control vs treatment
- guardrail metrics
- sample size
- statistical significance
- ramp strategy
- rollback path

If you say "I would A/B test it" and stop there, that sounds junior.

If you say:

> "I would define the primary metric, guardrails, minimum detectable effect, and ramp plan"

that sounds like someone trusted with production.

---

# Quick Thought Experiment

You built a fraud model with:

- 99.4% accuracy
- poor recall
- decent precision

Would you ship?

Only if the business is okay missing fraud.
Which is a poetic way of saying:

No.

---

# How Would You Deploy This Using Azure Pipelines?

Imagine your evaluation gate as a release check.

Before deployment, your pipeline should verify:

- model artifact version
- feature schema match
- validation metric threshold
- drift check against recent data
- latency benchmark
- rollback-ready previous model version

That mindset will instantly make your ML answers stronger.
