# Statistics and Probability

This is the fast stats file for interviews.

You do not need to become a probability opera singer.

You do need to explain the main ideas clearly and not misuse p-values with great confidence.

---

## 1. Core Distributions

- Bernoulli = one binary event
- Binomial = number of successes in fixed trials
- Poisson = event count in interval
- Exponential = time between Poisson events
- Normal = bell curve, everywhere

If you know when to use each, you are in good shape.

---

## 2. Mean, Variance, Standard Deviation

- mean = average
- variance = spread in squared units
- standard deviation = spread in original units

Variance is math-friendly.
Standard deviation is people-friendly.

---

## 3. Correlation vs Causation

Correlation means variables move together.
Causation means one changes the other.

Predictive ML often uses correlation well.
Decision-making and intervention require more care.

That is a strong sentence to remember.

---

## 4. P-Value

A p-value is the probability of observing data this extreme or more, assuming the null hypothesis is true.

It is **not**:

- the probability the null is true
- the chance your idea is correct

That mistake is common and very fixable.

---

## 5. Confidence Interval

A confidence interval gives a plausible range for an estimate.

Very useful because it gives more information than a point estimate alone.

---

## 6. Type I vs Type II Error

- Type I = false positive
- Type II = false negative

Courtroom analogy still works:

- Type I = convict innocent
- Type II = free guilty

---

## 7. Bayes' Theorem

Bayes updates belief after seeing evidence.

That is the main idea.

It matters for:

- Naive Bayes
- Bayesian inference
- uncertainty-aware reasoning

---

## 8. MLE vs MAP

MLE:

- use data only

MAP:

- use data plus prior

Very useful ML connection:

regularization often looks like MAP with a prior.

---

## 9. CLT vs LLN

LLN:

- sample mean converges to true mean

CLT:

- sampling distribution of the mean becomes approximately normal

These get mixed up a lot, so getting them clean is a nice interview win.
