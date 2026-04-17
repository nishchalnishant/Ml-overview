# Probability and Statistics

This file is for the part of ML interviews where people suddenly become very interested in uncertainty, distributions, and whether you know the difference between "looks good" and "is actually supported by evidence."

Do not panic.

Think of stats as the quality-control layer for your intuition.

---

# 1. Why Probability and Statistics Matter in ML

Machine learning is full of uncertainty:

- noisy data
- imperfect labels
- sample bias
- random initialization
- fluctuating metrics

Statistics is what helps you reason through that uncertainty without bluffing.

If DevOps gives you observability for systems, stats gives you observability for data and decisions.

---

# 2. Normal Distribution

The normal distribution is the classic bell curve.

It is:

- continuous
- symmetric
- defined by mean and variance

Why it matters so much:

Because many aggregate phenomena behave approximately normally, especially when they are the combined result of many small random effects.

**Short interview answer**

The normal distribution is fundamental because many natural processes and many sampling-based estimates end up approximately normal, which makes it central to inference and modeling.

---

# 3. Common Distributions You Should Actually Remember

You do not need to memorize every obscure distribution ever invented.

Know these cleanly:

## Bernoulli

One binary event.

Example:

- click or no click

## Binomial

Number of successes in multiple Bernoulli trials.

Example:

- number of conversions out of 100 visitors

## Poisson

Count of events in a fixed interval.

Example:

- number of requests per minute

## Exponential

Time between Poisson events.

Example:

- time until the next request arrives

## Normal

Continuous bell curve.

Example:

- noise around sensor measurements

**Mini memory trick**

- Bernoulli = one shot
- Binomial = many shots
- Poisson = event count
- Exponential = waiting time

---

# 4. Poisson vs Binomial

This gets asked more often than people expect.

## Binomial

Use when:

- fixed number of trials
- each trial has success/failure

## Poisson

Use when:

- counting events in time or space
- number of potential trials is not the natural framing

**Short answer**

Binomial is about successes out of a fixed number of trials; Poisson is about event counts over an interval.

---

# 5. Mean, Median, Mode

These are basic, but they matter because they reveal what kind of distribution you are dealing with.

## Mean

Average.

Sensitive to outliers.

## Median

Middle value.

More robust under skew or extreme values.

## Mode

Most frequent value.

Useful for:

- categorical data
- highly repeated values

**Fashion pricing analogy**

If most outfits cost 5k to 15k and one couture piece costs 4 lakhs, the mean can become dramatic very quickly.
The median usually stays calmer.

---

# 6. Variance and Standard Deviation

Variance measures how spread out values are around the mean.

Standard deviation is just the square root of variance.

Why both matter:

- variance is mathematically convenient
- standard deviation is easier to interpret

Think of them as:

- variance = the formula-friendly version
- standard deviation = the human-friendly version

---

# 7. Correlation vs Covariance

## Covariance

Measures whether two variables move together.

Problem:

It depends on units, so the raw number can be hard to interpret.

## Correlation

Normalized version of covariance.

Much easier to compare across feature pairs because it stays between:

- -1
- 0
- 1

**Important caveat**

Correlation measures linear relationship.
Zero correlation does not mean no relationship.

---

# 8. Correlation vs Causation

This is one of the most important sanity checks in data work.

Just because two things move together does not mean one causes the other.

Possible reasons for correlation:

- confounder
- reverse causality
- coincidence
- selection bias

**Short interview line**

Predictive ML often benefits from correlation, but decision-making and intervention require causal thinking.

That is a very strong sentence.

---

# 9. Law of Large Numbers vs Central Limit Theorem

These two get mixed up a lot.

## Law of Large Numbers

As sample size grows, the sample mean gets closer to the true population mean.

## Central Limit Theorem

As sample size grows, the sampling distribution of the mean becomes approximately normal, under broad conditions.

**Easy memory trick**

- LLN = estimate stabilizes
- CLT = sampling distribution becomes normal

---

# 10. P-Value

A p-value is the probability of observing a result at least as extreme as the one you got, assuming the null hypothesis is true.

That is it.

Not:

- the probability the null is true
- the probability your idea is correct

**Short answer**

A p-value tells you how surprising the observed data would be if there were actually no effect.

---

# 11. Statistical Significance

If the p-value is below a threshold like 0.05, people often say the result is statistically significant.

But be careful.

Statistical significance does **not** guarantee:

- practical importance
- causality
- production value

This is where many interview answers become weak.

Do not stop at:

> "p < 0.05"

Talk about:

- effect size
- confidence intervals
- business importance

That is the stronger answer.

---

# 12. Type I and Type II Errors

## Type I Error

False positive.

You think there is an effect when there is not.

## Type II Error

False negative.

You miss a real effect.

**Courtroom analogy**

- Type I = convicting an innocent person
- Type II = letting the guilty go free

Still the cleanest analogy.

---

# 13. Confidence Intervals

A confidence interval gives a range of plausible values for an estimate.

It is useful because it tells you more than a single point estimate.

It shows:

- uncertainty
- stability
- likely effect range

**Short interview answer**

Confidence intervals are often more informative than p-values alone because they communicate both uncertainty and effect size.

---

# 14. Bayes' Theorem

Bayes' theorem tells you how to update belief after seeing evidence.

Core idea:

- start with prior belief
- see new evidence
- update to posterior belief

This matters in ML because it sits under:

- Naive Bayes
- Bayesian inference
- uncertainty-aware modeling

**Short answer**

Bayes' theorem updates the probability of a hypothesis after observing evidence.

---

# 15. Naive Bayes

Naive Bayes uses Bayes' theorem and assumes features are conditionally independent given the class.

That assumption is clearly simplistic.

And yet it works surprisingly well in:

- text classification
- sparse feature settings
- small-data baselines

Why?

Because simple probabilistic structure can still be very effective.

---

# 16. MLE vs MAP

## MLE

Pick parameters that maximize likelihood of the observed data.

## MAP

Pick parameters that maximize posterior probability, meaning:

- likelihood
- plus prior belief

**Very useful ML connection**

Regularization often behaves like MAP estimation with a prior.

Examples:

- L2 regularization ~ Gaussian prior
- L1 regularization ~ Laplace prior

That is an elegant interview connection and worth remembering.

---

# 17. Bootstrap

Bootstrap is a resampling method.

You repeatedly sample with replacement from the observed data and compute the statistic again and again.

Why it is useful:

- estimate uncertainty
- estimate confidence intervals
- avoid relying on closed-form math when it is inconvenient

It is practical, intuitive, and very interview-worthy.

---

# 18. Statistical Tests for Comparing Models

The right test depends on context.

Examples:

- paired t-test for repeated performance comparisons when assumptions are okay
- McNemar's test for paired classification outcomes
- non-parametric alternatives when distribution assumptions are weak

**Good answer style**

Do not just name a test.
Say:

> "I would choose a test based on whether predictions are paired, whether assumptions hold, and whether I care about model outputs or business outcomes."

That sounds mature.

---

# Quick Thought Experiment

You run an A/B test on two ranking models.
Model B improves CTR by 0.2%.

Before celebrating, what do you ask?

- Is it statistically significant?
- Is the effect size meaningful?
- Did guardrail metrics worsen?
- Was the experiment powered enough?

That is how you keep data science from becoming astrology.

---

# Mini Pop Quiz

If I say:

> "The p-value is 0.03, so there is a 97% chance the model is better."

Is that correct?

No.

That is not what p-values mean.

And fixing that misunderstanding already puts you ahead of a surprising number of people.
