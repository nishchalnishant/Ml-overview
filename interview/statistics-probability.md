# Statistics & Probability: 30+ Questions

---

## 📊 Probability Fundamentals

**1. What is Bayes' Theorem?**
> $P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}$. Updates prior belief given new evidence.

**2. Give a practical example of Bayes' Theorem.**
> Disease testing: If 1% have disease, test is 90% accurate, 5% false positive. Positive result → only ~15% chance of disease due to low base rate.

**3. What is the difference between Prior and Posterior?**
> **Prior**: Belief before seeing data. **Posterior**: Updated belief after seeing data.

**4. What is Conditional Probability?**
> $P(A|B)$ = Probability of A given B has occurred.

**5. What is the Law of Total Probability?**
> $P(A) = \sum_i P(A|B_i)P(B_i)$ where $B_i$ partition the sample space.

**6. What is Independence?**
> Events A and B are independent if $P(A \cap B) = P(A) \cdot P(B)$.

**7. What is Marginal Probability?**
> Probability of a single event regardless of other variables: $P(A) = \sum_B P(A,B)$.

---

## 📈 Distributions

**8. Describe the Normal (Gaussian) distribution.**
> Bell curve. Defined by mean $\mu$ and std $\sigma$. 68-95-99.7 rule.

**9. What is the Central Limit Theorem?**
> Sum/mean of many independent random variables approaches Normal, regardless of original distribution.

**10. What is a Bernoulli distribution?**
> Single binary trial with probability p. $E[X] = p$, $Var(X) = p(1-p)$.

**11. What is a Binomial distribution?**
> Number of successes in n independent Bernoulli trials. $E[X] = np$.

**12. What is a Poisson distribution?**
> Models rare events in fixed interval. $P(X=k) = \frac{\lambda^k e^{-\lambda}}{k!}$.

**13. When do you use Poisson?**
> Count data: clicks per minute, errors per hour, arrivals per day.

**14. What is an Exponential distribution?**
> Time between Poisson events. Memoryless property.

**15. What is a Uniform distribution?**
> All outcomes equally likely. Continuous or discrete.

**16. What is a Beta distribution?**
> Distribution over probabilities (0,1). Prior for Bernoulli. Params: $\alpha$, $\beta$.

---

## 🧪 Hypothesis Testing

**17. What is a Null Hypothesis?**
> Default assumption of no effect. $H_0$: "Model A = Model B".

**18. What is a p-value?**
> Probability of observing data (or more extreme) if null hypothesis is true.

**19. What does p < 0.05 mean?**
> Less than 5% chance of seeing this result if null is true. Reject $H_0$.

**20. What is Type I Error?**
> False Positive. Rejecting $H_0$ when it's true. Controlled by $\alpha$ (significance level).

**21. What is Type II Error?**
> False Negative. Failing to reject $H_0$ when it's false. Related to power.

**22. What is Statistical Power?**
> Probability of correctly rejecting false $H_0$. Power = 1 - Type II error rate.

**23. What is a Confidence Interval?**
> Range likely to contain true parameter. 95% CI: If repeated 100 times, ~95 would contain truth.

**24. Difference between p-value and confidence interval?**
> **p-value**: Probability of data given $H_0$. **CI**: Range of plausible parameter values.

**25. What is the t-test?**
> Compares means of two groups. Assumes normally distributed data.

**26. What is the Chi-squared test?**
> Tests independence between categorical variables.

**27. What is ANOVA?**
> Compares means across 3+ groups. Extension of t-test.

---

## 📐 Statistical Concepts for ML

**28. What is Expectation?**
> $E[X] = \sum x_i P(x_i)$. The average value.

**29. What is Variance?**
> $Var(X) = E[(X - E[X])^2]$. Measures spread.

**30. What is Standard Deviation?**
> $\sigma = \sqrt{Var(X)}$. Same units as data.

**31. What is Covariance?**
> $Cov(X,Y) = E[(X-\mu_X)(Y-\mu_Y)]$. How X and Y change together.

**32. What is Correlation?**
> $\rho = \frac{Cov(X,Y)}{\sigma_X \sigma_Y}$. Normalized covariance. Range [-1, 1].

**33. Correlation vs Causation?**
> Correlation = statistical relationship. Causation = one causes the other. Correlation ≠ causation.

**34. What is MLE (Maximum Likelihood Estimation)?**
> Find parameter $\theta$ that maximizes probability of observed data: $\hat{\theta} = \arg\max \prod P(x_i|\theta)$.

**35. Derive MLE for Gaussian mean.**
> Log-likelihood → derivative w.r.t. $\mu$ → set to 0 → $\hat{\mu} = \frac{1}{n}\sum x_i$ (sample mean).

**36. What is MAP (Maximum A Posteriori)?**
> Like MLE but includes prior: $\hat{\theta} = \arg\max P(\theta|X) \propto P(X|\theta)P(\theta)$.

**37. What is the Law of Large Numbers?**
> Sample mean converges to population mean as sample size increases.

---

## 🎲 A/B Testing & Experimentation

**38. How do you determine sample size for A/B test?**
> Based on: effect size, baseline rate, significance level ($\alpha$), power ($1-\beta$). Use power analysis.

**39. What is the Minimum Detectable Effect (MDE)?**
> Smallest effect size you can reliably detect with given sample size and power.

**40. Why is "peeking" at A/B results bad?**
> Inflates false positive rate. You'll stop early on noise. Use sequential testing if you need to peek.

**41. What is Multiple Testing Correction?**
> When running many tests, false positives accumulate. Use Bonferroni or FDR correction.

**42. What is Bootstrapping?**
> Resample with replacement to estimate statistics. Useful for CIs without assumptions.

**43. What is the difference between Parametric and Non-parametric tests?**
> **Parametric**: Assumes distribution (t-test). **Non-parametric**: No assumptions (Mann-Whitney).

**44. What is Simpson's Paradox?**
> Trend appears in subgroups but reverses when combined. Always segment your analysis.

**45. What is Selection Bias?**
> Sample is not representative of population. Invalidates conclusions.
