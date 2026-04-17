# Probability and Statistics

In interviews, statistics answers are strongest when they connect a definition to modeling, uncertainty, or evaluation. The goal is to sound operational, not purely textbook.

---

# Q1: Explain the Bias-Variance Tradeoff.

**Interview-ready answer**

Bias is error from overly simple assumptions, while variance is error from being too sensitive to the training data. High-bias models underfit and miss important structure; high-variance models overfit and chase noise. The tradeoff is that increasing model flexibility often reduces bias but increases variance, so the best model is the one that minimizes expected generalization error, not training error.

---

# Q2: Explain different probability distributions (Normal, Binomial, Poisson, Uniform).

**Interview-ready answer**

The normal distribution models continuous values clustered around a mean with symmetric spread. The binomial models the number of successes in a fixed number of independent Bernoulli trials. The Poisson models the count of events occurring in a fixed interval when events happen independently at some average rate. The uniform distribution assigns equal probability density or mass over a specified range. In interviews, the key is not only naming them but explaining when each is a reasonable approximation.

---

# Q3: What is the normal distribution and its functions?

**Interview-ready answer**

The normal distribution is a continuous bell-shaped distribution defined by its mean and variance. It is important because many natural aggregate phenomena are approximately normal, and because the central limit theorem makes sample means tend toward normality under broad conditions. Its density function describes pointwise likelihood, its CDF gives cumulative probability, and z-scores standardize values relative to the mean and standard deviation.

---

# Q4-Q11: Common distributions (Exponential, Binomial, Bernoulli, Multinomial, Log-normal, Logistic, Gamma, Poisson)

**Interview-ready answer**

Bernoulli models a single yes-no outcome, and binomial sums repeated Bernoulli trials. Multinomial generalizes that to more than two categories. Poisson models event counts in time or space, while the exponential models waiting time between Poisson events. Gamma is a flexible positive-valued distribution often used for waiting times or rates. Log-normal models positive variables whose logarithm is normally distributed, which is common for skewed quantities like income or latency. Logistic is similar in shape to the normal but has heavier tails and appears naturally in logistic regression through the sigmoid link.

**Good interview framing**

If asked to compare them, anchor the answer on support and use case: binary, counts, waiting time, or positive skewed continuous values.

---

# Q12: When would you use a Poisson distribution over a Binomial distribution?

**Interview-ready answer**

I would use a Poisson distribution when I am modeling the number of events in a fixed interval and I care about an average rate rather than a fixed known number of trials. I would use a binomial distribution when there is a clear number of independent trials with a success probability per trial. A common approximation is that a binomial with large `n` and small `p` can be approximated by a Poisson with `lambda = np`.

---

# Q13-Q14: Variance and Stddev

**Interview-ready answer**

Variance measures the average squared deviation from the mean, while standard deviation is the square root of variance and is therefore in the same units as the data. Variance is mathematically convenient and appears naturally in many derivations, but standard deviation is usually easier to interpret. In ML, both show up in normalization, uncertainty estimation, and the bias-variance tradeoff.

---

# Q15: Explain mean, median, and mode.

**Interview-ready answer**

The mean is the arithmetic average, the median is the middle value when data is ordered, and the mode is the most frequent value. The main interview point is robustness: the mean is sensitive to outliers, the median is more stable under skew, and the mode is useful mainly for categorical data or strongly multi-modal distributions.

---

# Q16: Correlation vs covariance.

**Interview-ready answer**

Covariance measures how two variables vary together, but its magnitude depends on the units of the variables. Correlation is the normalized version of covariance, so it lies between `-1` and `1` and is unitless. That makes correlation easier to compare across pairs of variables, while covariance is often more useful in matrix form, such as the covariance matrix used in PCA.

---

# Q17: Correlation +1, 0, -1.

**Interview-ready answer**

A correlation of `+1` means a perfect positive linear relationship, `-1` means a perfect negative linear relationship, and `0` means no linear relationship. The important qualifier is "linear." A correlation of zero does not imply independence and does not rule out a strong non-linear relationship.

---

# Q18: Correlation vs Causation.

**Interview-ready answer**

Correlation means two variables move together; causation means changing one variable changes the other. Correlation can arise from confounding variables, reverse causality, selection bias, or coincidence. In ML interviews, the strongest answer is usually that predictive models often exploit correlation successfully, but causal decisions such as policy interventions require much stronger assumptions and experimental or quasi-experimental design.

---

# Q19-Q20: Type I / II errors

**Interview-ready answer**

A Type I error is a false positive: rejecting a null hypothesis that is actually true. A Type II error is a false negative: failing to reject a null hypothesis that is false. The tradeoff matters because lowering one often raises the other unless you collect more data or improve the test. In product language, this is closely related to precision-recall tradeoffs in classification.

---

# Q21-Q24: p-value, significance, limitations, hypothesis testing in ML

**Interview-ready answer**

A p-value is the probability, under the null hypothesis, of observing data at least as extreme as what you saw. Statistical significance means the result would be unlikely under the null at a chosen threshold, but it does not measure effect size, practical importance, or the probability that the hypothesis is true. In ML, hypothesis testing can be useful for model comparison, feature analysis, and experiments, but the answer should include limitations such as multiple testing, dependence violations, and the fact that predictive value is not guaranteed by statistical significance alone.

---

# Q25: What statistical tests would you use to compare two models?

**Interview-ready answer**

The right test depends on the setting. For paired predictions on the same classification examples, McNemar's test is common. For comparing continuous errors across folds or repeated runs, a paired t-test or non-parametric alternative like Wilcoxon signed-rank may be appropriate if assumptions hold. For online experiments, I would use standard A/B testing methods with confidence intervals and power analysis. The important part is to respect pairing and dependency structure rather than treating model outputs as independent observations.

---

# Q26: How do you assess if a feature is statistically significant?

**Interview-ready answer**

In classical models such as linear or logistic regression, I would look at coefficient estimates, standard errors, p-values, and confidence intervals, while also checking for multicollinearity and model misspecification. But statistical significance alone is not enough. A feature can be statistically significant and still have little predictive value or practical relevance. So I pair classical inference with ablation studies, validation performance, and domain sense.

---

# Q27: Confidence interval and usage.

**Interview-ready answer**

A confidence interval gives a range of plausible values for a parameter estimate under repeated sampling. For example, a 95 percent confidence interval is constructed so that the procedure would contain the true parameter 95 percent of the time across repeated samples. In interviews, a good answer emphasizes that confidence intervals communicate uncertainty and effect size better than p-values alone.

---

# Q28-Q29: z-score and t-score

**Interview-ready answer**

A z-score standardizes a value using the population mean and standard deviation, while a t-score is used when the population standard deviation is unknown and must be estimated from sample data. The t-distribution has heavier tails, which reflects extra uncertainty in small samples. In practice, use z-based reasoning when sample sizes are large or population variance is known, and t-based reasoning when estimating variance from limited data.

---

# Q30: Bayes' Theorem and Naive Bayes / Bayesian methods.

**Interview-ready answer**

Bayes' theorem updates prior beliefs using observed evidence: posterior is proportional to likelihood times prior. Naive Bayes applies this by assuming conditional independence of features given the class, which makes the posterior easy to compute and often works well in high-dimensional sparse problems like text. More generally, Bayesian methods treat parameters as random variables and produce posterior distributions rather than single point estimates, which is valuable when uncertainty matters.

---

# Q31: MLE vs MAP.

**Interview-ready answer**

Maximum likelihood estimation chooses the parameter values that make the observed data most likely. Maximum a posteriori estimation adds a prior and chooses the parameter values that maximize the posterior instead. So MLE uses data only, while MAP combines data with prior beliefs or regularization. In many common cases, MAP corresponds to MLE plus a regularization term.

---

# Q32: What is Maximum Likelihood Estimation (MLE)?

**Interview-ready answer**

MLE is a principle for estimating model parameters by choosing the values that maximize the likelihood of the observed data under the assumed model. It is foundational because many standard ML losses come directly from likelihood assumptions. For example, squared error corresponds to a Gaussian noise assumption and cross-entropy corresponds to Bernoulli or categorical likelihood.

---

# Q33: Bayesian vs Frequentist; CLT; sampling; bootstrap (combined essentials)

**Interview-ready answer**

Frequentist statistics treats parameters as fixed and data as random, while Bayesian statistics treats parameters as uncertain and updates beliefs through the posterior. The central limit theorem says that under broad conditions, sample means become approximately normal as sample size grows, which is why normal approximations appear so often. Sampling determines whether your data is representative and therefore whether your conclusions generalize. Bootstrap is a resampling method that estimates uncertainty by repeatedly sampling with replacement from the observed dataset, which is especially useful when analytic variance formulas are inconvenient.

**Good closing line**

If you want one unifying sentence: statistics in ML is about estimating signal and uncertainty under imperfect data and assumptions.
