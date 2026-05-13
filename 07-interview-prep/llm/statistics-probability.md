# Statistics and Probability — Interview Reference

---

## 1. Core Distributions

| Distribution | PMF / PDF | Mean | Variance | Use in ML |
| :--- | :--- | :--- | :--- | :--- |
| Bernoulli$(p)$ | $P(X=1)=p$ | $p$ | $p(1-p)$ | Binary classification output |
| Binomial$(n,p)$ | $\binom{n}{k}p^k(1-p)^{n-k}$ | $np$ | $np(1-p)$ | Number of correct predictions |
| Poisson$(\lambda)$ | $e^{-\lambda}\lambda^k/k!$ | $\lambda$ | $\lambda$ | Event counts (clicks, errors) |
| Exponential$(\lambda)$ | $\lambda e^{-\lambda x}$ | $1/\lambda$ | $1/\lambda^2$ | Time between events |
| Normal$(\mu,\sigma^2)$ | $\frac{1}{\sigma\sqrt{2\pi}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}$ | $\mu$ | $\sigma^2$ | Residuals, weight init |

**Gaussian properties critical for ML:**
- Sum of independent Gaussians is Gaussian: $X+Y \sim \mathcal{N}(\mu_X+\mu_Y, \sigma_X^2+\sigma_Y^2)$
- 68-95-99.7 rule: $P(\mu \pm \sigma) \approx 68\%$, $P(\mu \pm 2\sigma) \approx 95\%$, $P(\mu \pm 3\sigma) \approx 99.7\%$

---

## 2. Descriptive Statistics

$$\text{Mean: } \bar{x} = \frac{1}{n}\sum_{i=1}^n x_i$$

$$\text{Variance: } s^2 = \frac{1}{n-1}\sum_{i=1}^n (x_i - \bar{x})^2 \quad \text{(Bessel's correction: } n-1 \text{ for unbiased estimate)}$$

$$\text{Standard deviation: } s = \sqrt{s^2}$$

$$\text{Covariance: } \text{Cov}(X,Y) = \frac{1}{n-1}\sum(x_i - \bar{x})(y_i - \bar{y})$$

$$\text{Pearson correlation: } r = \frac{\text{Cov}(X,Y)}{s_X s_Y} \in [-1, 1]$$

**Correlation vs causation:** correlation measures linear co-movement. Causation requires an intervention. ML models exploit correlation for prediction — do not confuse with causal claims.

---

## 3. Bayes' Theorem

$$P(A \mid B) = \frac{P(B \mid A) \cdot P(A)}{P(B)}$$

$$\underbrace{P(\theta \mid D)}_{\text{posterior}} = \frac{\underbrace{P(D \mid \theta)}_{\text{likelihood}} \cdot \underbrace{P(\theta)}_{\text{prior}}}{\underbrace{P(D)}_{\text{normalizer}}}$$

**Practical example:** spam detection
- $P(\text{spam}) = 0.2$ — prior
- $P(\text{"free"} \mid \text{spam}) = 0.8$ — likelihood
- $P(\text{"free"}) = 0.2 \times 0.8 + 0.8 \times 0.05 = 0.2$
- $P(\text{spam} \mid \text{"free"}) = 0.8 \times 0.2 / 0.2 = 0.8$

**Naive Bayes:** assumes features are conditionally independent given class:
$$P(y \mid x_1, \ldots, x_n) \propto P(y) \prod_{i=1}^n P(x_i \mid y)$$

---

## 4. MLE vs MAP

**MLE (Maximum Likelihood Estimation):**
$$\hat{\theta}_{\text{MLE}} = \arg\max_\theta P(D \mid \theta) = \arg\max_\theta \sum_i \log P(x_i \mid \theta)$$

**MAP (Maximum A Posteriori):**
$$\hat{\theta}_{\text{MAP}} = \arg\max_\theta P(\theta \mid D) = \arg\max_\theta [\log P(D \mid \theta) + \log P(\theta)]$$

**Connection to regularization:**
- Gaussian prior on $\theta$: $P(\theta) \propto e^{-\lambda\|\theta\|^2}$ → MAP = MLE + L2 regularization (Ridge)
- Laplace prior on $\theta$: $P(\theta) \propto e^{-\lambda|\theta|}$ → MAP = MLE + L1 regularization (Lasso)

This is why regularization has a probabilistic interpretation.

---

## 5. Hypothesis Testing

**P-value:** probability of observing data this extreme or more extreme, *assuming the null hypothesis is true*.

$$p = P(\text{test statistic} \geq t_{\text{obs}} \mid H_0)$$

**What a p-value is NOT:**
- Not $P(H_0 \text{ is true})$
- Not $P(\text{your result is a fluke})$
- Not a measure of effect size

**Errors:**

| | Predict $H_0$ | Predict $H_1$ |
| :--- | :--- | :--- |
| **True $H_0$** | Correct ✓ | **Type I error** $(\alpha)$ |
| **True $H_1$** | **Type II error** $(\beta)$ | Correct ✓ |

- $\alpha$ = significance level (false positive rate), typically 0.05
- $\beta$ = false negative rate
- **Power** = $1 - \beta$ = probability of detecting a true effect

**Z-test for proportions** (A/B testing):
$$z = \frac{\hat{p}_1 - \hat{p}_2}{\sqrt{\hat{p}(1-\hat{p})(\frac{1}{n_1}+\frac{1}{n_2})}}, \quad \hat{p} = \frac{x_1+x_2}{n_1+n_2}$$

---

## 6. Confidence Intervals

$$\text{CI} = \bar{x} \pm z_{\alpha/2} \cdot \frac{s}{\sqrt{n}}$$

For 95% CI: $z_{0.025} = 1.96$.

**Correct interpretation:** if we repeat this experiment many times, 95% of the intervals constructed this way would contain the true parameter.

**Incorrect interpretation:** "there is a 95% probability the true value is in this specific interval."

---

## 7. Central Limit Theorem and Law of Large Numbers

**LLN:** as $n \to \infty$, the sample mean converges to the true mean:
$$\bar{X}_n \xrightarrow{p} \mu$$

**CLT:** the sampling distribution of the mean is approximately normal, regardless of the original distribution:
$$\frac{\bar{X}_n - \mu}{\sigma/\sqrt{n}} \xrightarrow{d} \mathcal{N}(0,1) \quad \text{as } n \to \infty$$

**Why CLT matters for ML:** justifies treating model performance metrics (averages over a test set) as approximately normally distributed, enabling standard error calculations and confidence intervals.

---

## 8. Probability Rules

**Addition rule:**
$$P(A \cup B) = P(A) + P(B) - P(A \cap B)$$

**Multiplication rule:**
$$P(A \cap B) = P(A \mid B) \cdot P(B)$$

**Independence:** $A \perp B$ iff $P(A \cap B) = P(A) \cdot P(B)$

**Total probability:**
$$P(B) = \sum_i P(B \mid A_i) P(A_i)$$

---

## 9. Information Theory

**Entropy** (average information content):
$$H(X) = -\sum_x P(x) \log_2 P(x) \quad \text{(bits)}$$

Maximum entropy = uniform distribution. Zero entropy = deterministic.

**Cross-entropy** (used as loss in classification):
$$H(p, q) = -\sum_x p(x) \log q(x)$$

where $p$ is the true distribution and $q$ is the model's predicted distribution.

**KL divergence** (how much $q$ differs from $p$):
$$D_{\text{KL}}(p \| q) = \sum_x p(x) \log \frac{p(x)}{q(x)} = H(p,q) - H(p)$$

$D_{\text{KL}} \geq 0$, equals 0 iff $p = q$. Not symmetric: $D_{\text{KL}}(p\|q) \neq D_{\text{KL}}(q\|p)$.

**Connection:** cross-entropy loss = $H(p,q)$ = KL divergence from true labels + constant. Minimizing cross-entropy loss = minimizing KL divergence to the true distribution.

---

## 10. A/B Testing for ML

**Sample size calculation** (two-proportion z-test):
$$n = \frac{(z_{\alpha/2} + z_\beta)^2 \cdot 2\bar{p}(1-\bar{p})}{\delta^2}$$

where $\delta$ is the minimum detectable effect, $\bar{p}$ is baseline rate.

**Common pitfalls:**
- **Peeking:** stopping early when $p < 0.05$ inflates Type I error — use sequential testing
- **Multiple comparisons:** testing 20 metrics at $\alpha=0.05$ → 1 false positive expected — use Bonferroni or FDR correction
- **Simpson's paradox:** aggregated data shows the opposite trend from sub-groups — always check for confounders

```python
from scipy import stats

def ab_test(control_conversions, control_n, treatment_conversions, treatment_n, alpha=0.05):
    p_control = control_conversions / control_n
    p_treatment = treatment_conversions / treatment_n
    p_pool = (control_conversions + treatment_conversions) / (control_n + treatment_n)

    se = (p_pool * (1 - p_pool) * (1/control_n + 1/treatment_n)) ** 0.5
    z_stat = (p_treatment - p_control) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))  # two-tailed

    return {
        "lift": (p_treatment - p_control) / p_control,
        "z_stat": z_stat,
        "p_value": p_value,
        "significant": p_value < alpha,
    }
```

> [!TIP]
> **Interview structure:** Stats = three layers: (1) descriptive (mean/variance/correlation), (2) inferential (p-values, CIs, hypothesis tests — and their common misinterpretations), (3) probabilistic (Bayes, MLE/MAP → ties to regularization). The most impressive thing is knowing what p-values do NOT mean and connecting MAP estimation to regularization.
